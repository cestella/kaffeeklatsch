package com.caseystella.llm.embeddings;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.caseystella.llm.DownloadUtil;
import com.caseystella.llm.ModelHome;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public enum CrossEncoders {
  MS_MARCO_MiniLM_L6_v2(
      "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "https://huggingface.co/metarank/ce-msmarco-MiniLM-L6-v2/resolve/main/",
      new String[] {
          "tokenizer.json",
          "pytorch_model.onnx",
          "config.json",
          "tokenizer.json",
          "vocab.txt"
      }
  );

  String name;
  String[] fileURLs;

  CrossEncoders(String name, String baseURL, String[] modelFiles) {
    this.name = EncodingUtil.normalizeEmbeddingName(name);
    fileURLs = new String[modelFiles.length];
    for(int i = 0; i < modelFiles.length;++i) {
      fileURLs[i] = DownloadUtil.joinURLParts(baseURL, modelFiles[i]);
    }
  }

  public HuggingFaceCrossEncoder create() throws IOException {
    return create(ModelHome.EMBEDDINGS.get());
  }

  public HuggingFaceCrossEncoder create(Path embeddingModelDir) throws IOException {
    Path modelDir = Paths.get(embeddingModelDir.toString(), name);
    DownloadUtil.downloadFiles(modelDir.toString(), fileURLs);
    return new HuggingFaceCrossEncoder(modelDir);
  }
}
