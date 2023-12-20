package com.caseystella.llm.embeddings;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.caseystella.llm.DownloadUtil;
import com.caseystella.llm.ModelHome;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public enum BiEncoders {
  ALL_MPNET_BASE_V2(
      "sentence-transformers_all-mpnet-base-v2",
      "https://huggingface.co/Xenova/all-mpnet-base-v2/resolve/main/",
      new String[] {
          "tokenizer.json",
          "onnx/model.onnx",
          "config.json",
          "special_tokens_map.json",
          "tokenizer_config.json",
          "vocab.txt"
      }
  );
  String name;
  String[] fileURLs;

  BiEncoders(String name, String baseURL, String[] modelFiles) {
    this.name = EncodingUtil.normalizeEmbeddingName(name);
    fileURLs = new String[modelFiles.length];
    for(int i = 0; i < modelFiles.length;++i) {
      fileURLs[i] = DownloadUtil.joinURLParts(baseURL, modelFiles[i]);
    }
  }


  public IBiEncoder create() throws IOException {
    return create(ModelHome.EMBEDDINGS.get());
  }

  public IBiEncoder create(Path embeddingModelDir) throws IOException {
    Path modelDir = Paths.get(embeddingModelDir.toString(), name);
    DownloadUtil.downloadFiles(modelDir.toString(), fileURLs);
    return new HuggingFaceBiEncoder(modelDir);
  }
}
