package com.caseystella.llm.embeddings;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.util.function.Function;

public class OnnxUtil {
  public static OnnxTensor computeInput(
      Encoding[] encodings,
      long[] shape,
      Function<Encoding, long[]> encodingToId,
      OrtEnvironment environment)
      throws OrtException {
    LongBuffer buffer =
        ByteBuffer.allocateDirect(encodings.length * encodings[0].getIds().length * Long.BYTES)
            .order(ByteOrder.nativeOrder())
            .asLongBuffer();

    for (Encoding encoding : encodings) {
      buffer.put(encodingToId.apply(encoding));
    }
    buffer.flip();
    return OnnxTensor.createTensor(environment, buffer, shape);
  }
}
