package com.caseystella.llm.embeddings;

public class EncodingUtil {
  public static String normalizeEmbeddingName(String name) {
    return name.replace("/", "_").replace(" ", "_");
  }
  public static double cosineSimilarity(float[] vectorA, float[] vectorB) {
    if (vectorA.length != vectorB.length) {
      throw new IllegalArgumentException("Vectors must be of the same length");
    }

    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (int i = 0; i < vectorA.length; i++) {
      dotProduct += vectorA[i] * vectorB[i];
      normA += Math.pow(vectorA[i], 2);
      normB += Math.pow(vectorB[i], 2);
    }

    if (normA == 0 || normB == 0) {
      // If either vector is zero, cosine similarity is not defined
      throw new IllegalArgumentException("Cosine similarity is not defined for zero vectors");
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}
