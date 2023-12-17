package com.caseystella.llm.embeddings;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;

public class HuggingFaceBiEncoder implements IBiEncoder {

  private HuggingFaceTokenizer tokenizer = null;
  private Path onnxLocation;

  public HuggingFaceBiEncoder(HuggingFaceTokenizer tokenizer, Path onnxLocation) {
    this.tokenizer = tokenizer;
    this.onnxLocation = onnxLocation;
  }

  public HuggingFaceBiEncoder(String modelName, Path onnxLocation) {
    this(HuggingFaceTokenizer.newInstance(modelName), onnxLocation);
  }

  @Override
  public Iterable<float[]> embed(String[] sentences, Optional<OrtEnvironment> environmentOpt)
      throws OrtException {
    Encoding[] encodings = tokenizer.batchEncode(sentences);
    OrtEnvironment environment = environmentOpt.orElse(OrtEnvironment.getEnvironment());

    OrtSession session =
        environment.createSession(onnxLocation.toString(), new OrtSession.SessionOptions());
    Set<String> inputParams = session.getInputNames();
    Map<String, OnnxTensor> inputs = new HashMap<>();
    long[] shape = {encodings.length, encodings[0].getAttentionMask().length};
    if (inputParams.contains("input_ids")) {
      inputs.put("input_ids", computeInput(encodings, shape, Encoding::getIds, environment));
    }
    if (inputParams.contains("attention_mask")) {
      inputs.put(
          "attention_mask",
          computeInput(encodings, shape, Encoding::getAttentionMask, environment));
    }
    if (inputParams.contains("token_type_ids")) {
      inputs.put(
          "token_type_ids", computeInput(encodings, shape, Encoding::getTypeIds, environment));
    }
    try (OrtSession.Result results = session.run(inputs)) {
      OnnxTensor outputTensor = (OnnxTensor) results.get(0);
      FloatBuffer outputBuffer = outputTensor.getFloatBuffer();
      int dimension = outputBuffer.capacity() / sentences.length;
      ArrayList<float[]> ret = new ArrayList<>(sentences.length);
      for (int i = 0; i < sentences.length; ++i) {
        float[] embedding = new float[dimension];
        outputBuffer.get(embedding);
        ret.add(embedding);
      }
      return ret;
    } finally {
      session.close();
      if (environmentOpt.isEmpty()) {
        environment.close();
      }
    }
  }

  private OnnxTensor computeInput(
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
