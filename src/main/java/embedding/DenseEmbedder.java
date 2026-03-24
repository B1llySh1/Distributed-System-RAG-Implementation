package embedding;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * DenseEmbedder uses DJL to embed passages with sentence-transformers/all-MiniLM-L6-v2.
 *
 * Input TSV:  pid \t passage_text
 * Output TSV: pid \t f1,f2,...,f384
 *
 * Usage:
 *   java -cp <classpath> embedding.DenseEmbedder --input passages.tsv --output dense-raw.tsv [--batch 32]
 */
public class DenseEmbedder {

    private static final int DEFAULT_BATCH_SIZE = 32;

    private final ZooModel<String[], float[][]> model;
    private final Predictor<String[], float[][]> predictor;

    public DenseEmbedder() throws ModelException, IOException {
        Criteria<String[], float[][]> criteria = Criteria.builder()
                .optApplication(Application.NLP.TEXT_EMBEDDING)
                .setTypes(String[].class, float[][].class)
                .optModelUrls("djl://ai.djl.huggingface/sentence-transformers/all-MiniLM-L6-v2")
                .optEngine("OnnxRuntime")
                .optProgress(new ProgressBar())
                .build();

        model = criteria.loadModel();
        predictor = model.newPredictor();
    }

    /**
     * Embed a batch of texts and return their embeddings.
     *
     * @param texts array of input texts
     * @return 2-D float array [texts.length][embeddingDim]
     */
    public float[][] embed(String[] texts) throws TranslateException {
        return predictor.predict(texts);
    }

    /**
     * Close the predictor and model to release resources.
     */
    public void close() {
        predictor.close();
        model.close();
    }

    /**
     * Format a float array as a comma-separated string.
     */
    private static String floatArrayToString(float[] arr) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(arr[i]);
        }
        return sb.toString();
    }

    public static void main(String[] args) throws Exception {
        String inputPath  = "";
        String outputPath = "";
        int batchSize     = DEFAULT_BATCH_SIZE;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--input":
                    inputPath = args[++i];
                    break;
                case "--output":
                    outputPath = args[++i];
                    break;
                case "--batch":
                    batchSize = Integer.parseInt(args[++i]);
                    break;
                default:
                    System.err.println("Unknown argument: " + args[i]);
            }
        }

        if (inputPath.isEmpty() || outputPath.isEmpty()) {
            System.err.println("Usage: DenseEmbedder --input <tsv> --output <tsv> [--batch <size>]");
            System.err.println("  Input TSV columns: pid \\t passage_text");
            System.err.println("  Output TSV columns: pid \\t f1,f2,...,f384");
            System.exit(1);
        }

        System.out.println("Loading DJL model (all-MiniLM-L6-v2)...");
        DenseEmbedder embedder = new DenseEmbedder();
        System.out.println("Model loaded successfully.");

        long totalPassages = 0;
        long startTime     = System.currentTimeMillis();

        try (BufferedReader reader = new BufferedReader(new FileReader(inputPath));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {

            List<String> batchPids  = new ArrayList<>(batchSize);
            List<String> batchTexts = new ArrayList<>(batchSize);

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;

                String[] parts = line.split("\t", 2);
                if (parts.length < 2) {
                    System.err.println("Skipping malformed line: " + line);
                    continue;
                }

                batchPids.add(parts[0].trim());
                batchTexts.add(parts[1].trim());

                if (batchTexts.size() >= batchSize) {
                    flushBatch(embedder, batchPids, batchTexts, writer);
                    totalPassages += batchTexts.size();
                    batchPids.clear();
                    batchTexts.clear();

                    if (totalPassages % 10000 == 0) {
                        long elapsed = (System.currentTimeMillis() - startTime) / 1000;
                        System.out.printf("Processed %d passages in %ds%n", totalPassages, elapsed);
                    }
                }
            }

            // Flush remaining batch
            if (!batchTexts.isEmpty()) {
                flushBatch(embedder, batchPids, batchTexts, writer);
                totalPassages += batchTexts.size();
            }
        }

        embedder.close();

        long elapsed = (System.currentTimeMillis() - startTime) / 1000;
        System.out.printf("Done. Embedded %d passages in %ds. Output: %s%n",
                totalPassages, elapsed, outputPath);
    }

    private static void flushBatch(
            DenseEmbedder embedder,
            List<String> pids,
            List<String> texts,
            BufferedWriter writer) throws TranslateException, IOException {

        String[] textArray = texts.toArray(new String[0]);
        float[][] embeddings = embedder.embed(textArray);

        for (int i = 0; i < pids.size(); i++) {
            writer.write(pids.get(i));
            writer.write('\t');
            writer.write(floatArrayToString(embeddings[i]));
            writer.newLine();
        }
    }
}
