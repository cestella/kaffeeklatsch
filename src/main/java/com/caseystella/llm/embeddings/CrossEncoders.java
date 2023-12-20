package com.caseystella.llm.embeddings;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.caseystella.llm.ModelHome;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public enum CrossEncoders {
  MS_MARCO_MiniLM_L6_v2(
      "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "https://huggingface.co/metarank/ce-msmarco-MiniLM-L6-v2/raw/main",
      "tokenizer.json",
      "pytorch_model.onnx");

  String name;
  String tokenizerURL;
  String modelURL;

  CrossEncoders(String name, String tokenizerURL, String modelURL) {
    this.name = name;
    this.tokenizerURL = tokenizerURL;
    this.modelURL = modelURL;
  }

  CrossEncoders(String name, String baseURL, String tokenizerName, String modelName) {
    this(name, baseURL + "/" + tokenizerName, baseURL + "/" + modelName);
  }

  public HuggingFaceCrossEncoder create() throws IOException {
    return create(ModelHome.EMBEDDINGS.get());
  }

  public HuggingFaceCrossEncoder create(Path embeddingModelDir) throws IOException {
    Path modelDir = Paths.get(embeddingModelDir.toString(), name);
    HuggingFaceTokenizer tokenizer =
        HuggingFaceUtil.resolveTokenizer(embeddingModelDir, name, tokenizerURL, modelURL);
    return new HuggingFaceCrossEncoder(tokenizer, modelDir);
  }
}
