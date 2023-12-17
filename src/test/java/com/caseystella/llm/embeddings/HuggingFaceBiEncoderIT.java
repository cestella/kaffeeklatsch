package com.caseystella.llm.embeddings;

import ai.onnxruntime.OrtEnvironment;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.Optional;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class HuggingFaceBiEncoderIT {

  private OrtEnvironment environment;
  private Path embeddingModelHome;

  @BeforeEach
  public void setup() {
    embeddingModelHome = Path.of(System.getProperty("embedding.model.home"));
    environment = OrtEnvironment.getEnvironment();
  }

  @AfterEach
  public void teardown() {
    if (environment != null) {
      environment.close();
    }
  }

  @Test
  public void testBiEncoder() throws Exception {
    IBiEncoder encoder = BiEncoders.ALL_MPNET_BASE_V2.create(embeddingModelHome);
    String[] sentences = new String[] {"Hello world", "Greetings planet"};
    Iterator<float[]> embeddingsIt = encoder.embed(sentences, Optional.of(environment)).iterator();
    float[] embedding1 = embeddingsIt.next();
    float[] embedding2 = embeddingsIt.next();
    Assertions.assertFalse(embeddingsIt.hasNext());
    double cosSim = EncodingUtil.cosineSimilarity(embedding1, embedding2);
    Assertions.assertTrue(cosSim >= 0.7 && cosSim < 1.0);
  }
}
