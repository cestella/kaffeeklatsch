package com.caseystella.llm.embeddings;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.caseystella.llm.ModelHome;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public enum BiEncoders {
  ALL_MPNET_BASE_V2(
      "sentence-transformers_all-mpnet-base-v2",
      "https://huggingface.co/Xenova/all-mpnet-base-v2/raw/main/",
      "tokenizer.json",
      "onnx/model.onnx");

  String name;
  String tokenizerURL;
  String modelURL;

  BiEncoders(String name, String tokenizerURL, String modelURL) {
    this.name = name;
    this.tokenizerURL = tokenizerURL;
    this.modelURL = modelURL;
  }

  BiEncoders(String name, String baseURL, String tokenizerName, String modelName) {
    this(name, baseURL + "/" + tokenizerName, baseURL + "/" + modelName);
  }

  public IBiEncoder create() throws IOException {
    return create(ModelHome.EMBEDDINGS.get());
  }

  public IBiEncoder create(Path embeddingModelDir) throws IOException {
    Path modelDir = Paths.get(embeddingModelDir.toString(), name);
    HuggingFaceTokenizer tokenizer =
        HuggingFaceUtil.resolveTokenizer(embeddingModelDir, name, tokenizerURL, modelURL);
    return new HuggingFaceBiEncoder(tokenizer, modelDir);
  }
}
