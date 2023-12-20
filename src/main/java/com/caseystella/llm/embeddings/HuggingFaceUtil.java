package com.caseystella.llm.embeddings;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.caseystella.llm.DownloadUtil;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class HuggingFaceUtil {

  public static HuggingFaceTokenizer resolveTokenizer(String modelName, Path onnxLocation) {
    try {
      return HuggingFaceTokenizer.newInstance(onnxLocation.toFile().getParentFile().toPath());
    } catch (IOException e) {
      return HuggingFaceTokenizer.newInstance(modelName);
    }
  }

  public static HuggingFaceTokenizer resolveTokenizer(
      Path embeddingModelDir, String modelName, String tokenizerURL, String modelURL)
      throws IOException {
    Path modelDir =
        Paths.get(embeddingModelDir.toString(), modelName.replace(" ", "_").replace("/", "_"));
    DownloadUtil.downloadFiles(modelDir.toString(), tokenizerURL, modelURL);
    return HuggingFaceTokenizer.newInstance(modelDir);
  }
}
