//
//  ContentView.swift
//  Deneme2PowerConsumption
//
//  Created by Ali Kemal on 18.12.25.
//

import SwiftUI
import CoreML
import Combine

enum InferencePhase {
    case idle
    case warmup
    case silence
    case process
    case completed
    
    var description: String {
        switch self {
        case .idle: return "Ready to start"
        case .warmup: return "Warming up..."
        case .silence: return "Cooling down..."
        case .process: return "Processing..."
        case .completed: return "Completed"
        }
    }
    
    var color: Color {
        switch self {
        case .idle: return .blue
        case .warmup: return .orange
        case .silence: return .purple
        case .process: return .green
        case .completed: return .gray
        }
    }
}

struct ModelInfo: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let url: URL
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    static func == (lhs: ModelInfo, rhs: ModelInfo) -> Bool {
        lhs.id == rhs.id
    }
}

@MainActor
final class InferenceManager: ObservableObject {
    @Published var currentPhase: InferencePhase = .idle
    @Published var progress: Double = 0.0
    @Published var timeRemaining: Double = 0.0
    @Published var inferenceCount: Int = 0
    @Published var availableModels: [ModelInfo] = []
    @Published var errorMessage: String?
    @Published var totalInferences: Int = 0
    @Published var averageInferenceTime: Double = 0.0
    @Published var currentRun: Int = 0
    @Published var totalRuns: Int = 1
    
    private var task: Task<Void, Never>?
    private var loadedModel: MLModel?
    private var inferenceTimes: [Double] = []
    
    // Phase durations as per requirements
    let warmupDuration: Double = 6.0      // 6 seconds
    let silenceDuration: Double = 12.0    // 12 seconds
    let processDuration: Double = 42.0    // 42 seconds
    
    init() {
        discoverModels()
    }
    
    func discoverModels() {
        var models: [ModelInfo] = []
        
        // Try to find models in the bundle
        if let bundleURL = Bundle.main.resourceURL {
            // Look for .mlmodelc files (compiled models)
            if let enumerator = FileManager.default.enumerator(at: bundleURL, 
                                                               includingPropertiesForKeys: [.isDirectoryKey],
                                                               options: [.skipsHiddenFiles]) {
                for case let fileURL as URL in enumerator {
                    // Check if it's a compiled model
                    if fileURL.pathExtension == "mlmodelc" {
                        let modelName = fileURL.deletingPathExtension().lastPathComponent
                        models.append(ModelInfo(name: modelName, url: fileURL))
                    }
                }
            }
            
            // Also look for .mlpackage files
            if let enumerator = FileManager.default.enumerator(at: bundleURL,
                                                               includingPropertiesForKeys: [.isDirectoryKey],
                                                               options: [.skipsHiddenFiles]) {
                for case let fileURL as URL in enumerator {
                    if fileURL.pathExtension == "mlpackage" {
                        let modelName = fileURL.deletingPathExtension().lastPathComponent
                        // Avoid duplicates
                        if !models.contains(where: { $0.name == modelName }) {
                            models.append(ModelInfo(name: modelName, url: fileURL))
                        }
                    }
                }
            }
        }
        
        // Sort models alphabetically
        availableModels = models.sorted { $0.name < $1.name }
        
        if availableModels.isEmpty {
            print("⚠️ No models found in bundle. Make sure your .mlmodel files are added to the target.")
        } else {
            print("✅ Found \(availableModels.count) model(s): \(availableModels.map { $0.name }.joined(separator: ", "))")
        }
    }
    
    func startInference(modelInfo: ModelInfo, numberOfRuns: Int) {
        // Cancel any existing task
        task?.cancel()
        
        // Reset state
        inferenceCount = 0
        progress = 0.0
        errorMessage = nil
        loadedModel = nil
        inferenceTimes = []
        totalInferences = 0
        averageInferenceTime = 0.0
        currentRun = 0
        totalRuns = numberOfRuns
        
        task = Task {
            await runMultipleTests(modelInfo: modelInfo, numberOfRuns: numberOfRuns)
        }
    }
    
    func stop() {
        task?.cancel()
        currentPhase = .idle
        progress = 0.0
        timeRemaining = 0.0
        loadedModel = nil
        inferenceTimes = []
        totalInferences = 0
        averageInferenceTime = 0.0
        currentRun = 0
        totalRuns = 1
    }
    
    private func runMultipleTests(modelInfo: ModelInfo, numberOfRuns: Int) async {
        print("🚀 Starting test suite: \(numberOfRuns) run(s)")
        
        // Load the model once at the beginning
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all // Use all available compute units (CPU, GPU, Neural Engine)
            
            print("Loading model: \(modelInfo.name) from \(modelInfo.url.path)")
            loadedModel = try await MLModel.load(contentsOf: modelInfo.url, configuration: config)
            print("✅ Model loaded successfully")
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            print("❌ Error loading model: \(error)")
            currentPhase = .idle
            return
        }
        
        // Run the test cycle multiple times
        for runIndex in 1...numberOfRuns {
            guard !Task.isCancelled else {
                currentPhase = .idle
                loadedModel = nil
                return
            }
            
            currentRun = runIndex
            print("📊 Starting run \(runIndex) of \(numberOfRuns)")
            
            await runSingleTestCycle(runIndex: runIndex, totalRuns: numberOfRuns)
            
            // Add a small pause between runs (except after the last run)
            if runIndex < numberOfRuns {
                print("⏸️ Pausing between runs...")
                try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 second pause
            }
        }
        
        // Completed all runs
        currentPhase = .completed
        progress = 1.0
        timeRemaining = 0.0
        totalInferences = inferenceCount
        
        // Calculate average inference time (only from process phases)
        if !inferenceTimes.isEmpty {
            averageInferenceTime = inferenceTimes.reduce(0, +) / Double(inferenceTimes.count)
        }
        
        loadedModel = nil
        print("✅ All tests completed. Total runs: \(numberOfRuns), Total inferences: \(totalInferences), Average time: \(String(format: "%.2f", averageInferenceTime))ms")
    }
    
    private func runSingleTestCycle(runIndex: Int, totalRuns: Int) async {
        // Calculate base progress for this run
        let runProgress = Double(runIndex - 1) / Double(totalRuns)
        let runProgressRange = 1.0 / Double(totalRuns)
        
        // Phase 1: Warmup (6 seconds)
        currentPhase = .warmup
        await runWarmupPhase(baseProgress: runProgress, progressRange: runProgressRange)
        
        guard !Task.isCancelled else { return }
        
        // Phase 2: Silence (12 seconds)
        currentPhase = .silence
        await runSilencePhase(baseProgress: runProgress, progressRange: runProgressRange)
        
        guard !Task.isCancelled else { return }
        
        // Phase 3: Process (42 seconds)
        currentPhase = .process
        await runProcessPhase(baseProgress: runProgress, progressRange: runProgressRange)
        
        print("✅ Run \(runIndex) completed")
    }
    
    private func runInferenceWorkflow(modelInfo: ModelInfo) async {
        // This method is no longer used - replaced by runMultipleTests and runSingleTestCycle
        // Keeping for reference or can be removed
    }
    
    // Phase 1: Warmup - Continuously run inference for 6 seconds
    private func runWarmupPhase(baseProgress: Double, progressRange: Double) async {
        let startTime = CFAbsoluteTimeGetCurrent()
        let targetDuration = warmupDuration
        
        print("🔥 Starting warmup phase (\(warmupDuration)s)")
        
        while true {
            guard !Task.isCancelled else { return }
            
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            
            if elapsed >= targetDuration {
                break
            }
            
            // Run inference continuously
            _ = await runSingleInference()
            inferenceCount += 1
            
            // Update progress (warmup is 0-10% of each run)
            let phaseProgress = elapsed / targetDuration
            progress = baseProgress + (phaseProgress * 0.1 * progressRange)
            timeRemaining = targetDuration - elapsed
        }
        
        print("✅ Warmup complete. Inferences: \(inferenceCount)")
    }
    
    // Phase 2: Silence - Sleep for 100ms intervals for 12 seconds total
    private func runSilencePhase(baseProgress: Double, progressRange: Double) async {
        let startTime = CFAbsoluteTimeGetCurrent()
        let targetDuration = silenceDuration
        let sleepInterval: UInt64 = 100_000_000 // 100ms in nanoseconds
        
        print("🤫 Starting silence phase (\(silenceDuration)s)")
        
        while true {
            guard !Task.isCancelled else { return }
            
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            
            if elapsed >= targetDuration {
                break
            }
            
            // Sleep for 100ms
            try? await Task.sleep(nanoseconds: sleepInterval)
            
            // Update progress (silence is 10-30% of each run)
            let phaseProgress = elapsed / targetDuration
            progress = baseProgress + (0.1 * progressRange) + (phaseProgress * 0.2 * progressRange)
            timeRemaining = targetDuration - elapsed
        }
        
        print("✅ Silence phase complete")
    }
    
    // Phase 3: Process - Continuously run inference for 42 seconds and gather data
    private func runProcessPhase(baseProgress: Double, progressRange: Double) async {
        let startTime = CFAbsoluteTimeGetCurrent()
        let targetDuration = processDuration
        
        // Don't reset inference times - accumulate across all runs
        let processPhaseStartCount = inferenceCount
        
        print("⚙️ Starting process phase (\(processDuration)s)")
        
        while true {
            guard !Task.isCancelled else { return }
            
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            
            if elapsed >= targetDuration {
                break
            }
            
            // Run inference and track timing
            let inferenceTime = await runSingleInference()
            inferenceTimes.append(inferenceTime)
            inferenceCount += 1
            
            // Update progress (process is 30-100% of each run)
            let phaseProgress = elapsed / targetDuration
            progress = baseProgress + (0.3 * progressRange) + (phaseProgress * 0.7 * progressRange)
            timeRemaining = targetDuration - elapsed
        }
        
        let processInferences = inferenceCount - processPhaseStartCount
        print("✅ Process phase complete. Inferences: \(processInferences)")
    }
    
    private func runSingleInference() async -> Double {
        guard let model = loadedModel else {
            print("❌ No model loaded")
            return 0.0
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            // Create input features based on the model's input description
            let inputDescription = model.modelDescription.inputDescriptionsByName
            
            // Create a provider with the required inputs
            let provider = try createInputProvider(for: inputDescription)
            
            // Run prediction
            _ = try await model.prediction(from: provider)
            
            // You can optionally process the output here if needed
            // let output = prediction.featureValue(for: "outputName")
            
        } catch {
            print("❌ Inference error: \(error.localizedDescription)")
            errorMessage = "Inference error: \(error.localizedDescription)"
        }
        
        // Calculate and return inference time in milliseconds
        let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime
        return elapsedTime * 1000.0
    }
    
    private func createInputProvider(for inputDescription: [String: MLFeatureDescription]) throws -> MLFeatureProvider {
        var inputFeatures: [String: MLFeatureValue] = [:]
        
        // Generate appropriate inputs based on the model's requirements
        for (name, description) in inputDescription {
            switch description.type {
            case .image:
                // Create a dummy image input
                if let imageConstraint = description.imageConstraint {
                    let width = imageConstraint.pixelsWide
                    let height = imageConstraint.pixelsHigh
                    
                    // Create a simple image buffer
                    if let image = createDummyImage(width: width, height: height) {
                        inputFeatures[name] = MLFeatureValue(pixelBuffer: image)
                    }
                }
                
            case .multiArray:
                // Create a dummy multiarray input
                if let constraint = description.multiArrayConstraint {
                    let shape = constraint.shape.map { $0.intValue }
                    let array = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
                    
                    // Fill with random values
                    for i in 0..<array.count {
                        array[i] = NSNumber(value: Double.random(in: 0...1))
                    }
                    
                    inputFeatures[name] = MLFeatureValue(multiArray: array)
                }
                
            case .double:
                inputFeatures[name] = MLFeatureValue(double: Double.random(in: 0...1))
                
            case .int64:
                inputFeatures[name] = MLFeatureValue(int64: Int64.random(in: 0...100))
                
            case .string:
                inputFeatures[name] = MLFeatureValue(string: "test")
                
            default:
                print("⚠️ Unsupported input type for \(name): \(description.type)")
            }
        }
        
        return try MLDictionaryFeatureProvider(dictionary: inputFeatures)
    }
    
    private func createDummyImage(width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                        width,
                                        height,
                                        kCVPixelFormatType_32BGRA,
                                        attrs,
                                        &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        // Fill with some data
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        if let baseAddress = CVPixelBufferGetBaseAddress(buffer) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
            let bufferSize = bytesPerRow * height
            memset(baseAddress, 128, bufferSize) // Fill with gray
        }
        
        return buffer
    }
}

struct ModelPickerView: View {
    let models: [ModelInfo]
    @Binding var selectedModel: ModelInfo?
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            List {
                ForEach(models) { model in
                    Button {
                        selectedModel = model
                        dismiss()
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(model.name)
                                    .foregroundStyle(.primary)
                                    .font(.body)
                                    .multilineTextAlignment(.leading)
                            }
                            Spacer()
                            if selectedModel?.id == model.id {
                                Image(systemName: "checkmark")
                                    .foregroundStyle(.blue)
                                    .fontWeight(.semibold)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Select Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var inferenceManager = InferenceManager()
    @State private var selectedModel: ModelInfo?
    @State private var showingModelPicker = false
    @State private var numberOfRuns: Int = 1
    
    let runOptions = [1, 2, 3, 4, 5, 10]
    
    var body: some View {
        VStack(spacing: 30) {
            // Title
            Text("ML Model Inference Testing")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            // Model Selection
            VStack(alignment: .leading, spacing: 10) {
                Text("Select Model")
                    .font(.headline)
                
                if inferenceManager.availableModels.isEmpty {
                    HStack {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundStyle(.orange)
                        Text("No models found. Add .mlmodel files to your project.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.orange.opacity(0.1))
                    )
                } else {
                    VStack(spacing: 8) {
                        // Custom dropdown button
                        Button {
                            // Button action handled by sheet
                        } label: {
                            HStack {
                                Text(selectedModel?.name ?? "Choose a model...")
                                    .foregroundStyle(selectedModel == nil ? .secondary : .primary)
                                    .lineLimit(1)
                                Spacer()
                                Image(systemName: "chevron.down")
                                    .foregroundStyle(.secondary)
                                    .font(.caption)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color(.systemGray6))
                            )
                        }
                        .disabled(inferenceManager.currentPhase == .warmup || 
                                 inferenceManager.currentPhase == .silence ||
                                 inferenceManager.currentPhase == .process)
                        .sheet(isPresented: .init(
                            get: { showingModelPicker },
                            set: { showingModelPicker = $0 }
                        )) {
                            ModelPickerView(
                                models: inferenceManager.availableModels,
                                selectedModel: $selectedModel
                            )
                        }
                        .simultaneousGesture(TapGesture().onEnded {
                            if inferenceManager.currentPhase != .warmup && 
                               inferenceManager.currentPhase != .silence &&
                               inferenceManager.currentPhase != .process {
                                showingModelPicker = true
                            }
                        })
                        
                        if let model = selectedModel {
                            Text("Selected: \(model.name)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                    }
                }
            }
            
            // Number of Runs Selection
            VStack(alignment: .leading, spacing: 10) {
                Text("Number of Runs")
                    .font(.headline)
                
                Menu {
                    ForEach(runOptions, id: \.self) { option in
                        Button {
                            numberOfRuns = option
                        } label: {
                            HStack {
                                Text("\(option) \(option == 1 ? "run" : "runs")")
                                if numberOfRuns == option {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                } label: {
                    HStack {
                        Text("\(numberOfRuns) \(numberOfRuns == 1 ? "run" : "runs")")
                            .foregroundStyle(.primary)
                        Spacer()
                        Image(systemName: "chevron.down")
                            .foregroundStyle(.secondary)
                            .font(.caption)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color(.systemGray6))
                    )
                }
                .disabled(inferenceManager.currentPhase == .warmup || 
                         inferenceManager.currentPhase == .silence ||
                         inferenceManager.currentPhase == .process)
                
                Text("Total duration: ~\(numberOfRuns * 60) seconds")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            // Error Message
            if let error = inferenceManager.errorMessage {
                HStack {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.red)
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding()
                .frame(maxWidth: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.red.opacity(0.1))
                )
            }
            
            // Status Display
            VStack(spacing: 15) {
                HStack {
                    Circle()
                        .fill(inferenceManager.currentPhase.color)
                        .frame(width: 12, height: 12)
                    
                    Text(inferenceManager.currentPhase.description)
                        .font(.title3)
                        .fontWeight(.medium)
                    
                    Spacer()
                    
                    // Show current run progress
                    if inferenceManager.currentRun > 0 && inferenceManager.currentPhase != .completed && inferenceManager.currentPhase != .idle {
                        Text("Run \(inferenceManager.currentRun)/\(inferenceManager.totalRuns)")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(
                                Capsule()
                                    .fill(Color(.systemGray5))
                            )
                    }
                }
                
                if inferenceManager.currentPhase == .warmup || 
                   inferenceManager.currentPhase == .silence ||
                   inferenceManager.currentPhase == .process {
                    VStack(spacing: 8) {
                        // Progress Bar
                        ProgressView(value: inferenceManager.progress)
                            .progressViewStyle(.linear)
                            .frame(height: 20)
                            .scaleEffect(x: 1, y: 2, anchor: .center)
                        
                        HStack {
                            Text("Progress: \(Int(inferenceManager.progress * 100))%")
                            Spacer()
                            Text("Time: \(String(format: "%.1f", inferenceManager.timeRemaining))s")
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        
                        Text("Inferences: \(inferenceManager.inferenceCount)")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
                
                // Completion Statistics
                if inferenceManager.currentPhase == .completed {
                    VStack(spacing: 12) {
                        Divider()
                        
                        VStack(spacing: 8) {
                            HStack {
                                Text("Completed Runs:")
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text("\(inferenceManager.totalRuns)")
                                    .fontWeight(.semibold)
                            }
                            
                            HStack {
                                Text("Total Inferences:")
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text("\(inferenceManager.totalInferences)")
                                    .fontWeight(.semibold)
                            }
                            
                            HStack {
                                Text("Average Time:")
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text(String(format: "%.2f ms", inferenceManager.averageInferenceTime))
                                    .fontWeight(.semibold)
                            }
                        }
                        .font(.subheadline)
                    }
                }
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(.systemBackground))
                    .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
            )
            
            // Control Buttons
            HStack(spacing: 20) {
                Button {
                    if let model = selectedModel {
                        inferenceManager.startInference(modelInfo: model, numberOfRuns: numberOfRuns)
                    }
                } label: {
                    Label("Start", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(selectedModel == nil || 
                         inferenceManager.currentPhase == .warmup || 
                         inferenceManager.currentPhase == .silence ||
                         inferenceManager.currentPhase == .process)
                
                Button {
                    inferenceManager.stop()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .tint(.red)
                .disabled(inferenceManager.currentPhase == .idle || 
                         inferenceManager.currentPhase == .completed)
            }
            
            Spacer()
        }
        .padding()
        .onAppear {
            // Set the first model as default if available
            if selectedModel == nil, let firstModel = inferenceManager.availableModels.first {
                selectedModel = firstModel
            }
        }
    }
}

#Preview {
    ContentView()
}
