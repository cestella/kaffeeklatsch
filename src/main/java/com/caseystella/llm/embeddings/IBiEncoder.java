package com.caseystella.llm.embeddings;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import java.util.Optional;

public interface IBiEncoder {
  Iterable<float[]> embed(String[] sentences, Optional<OrtEnvironment> environmentOpt)
      throws OrtException;
}
