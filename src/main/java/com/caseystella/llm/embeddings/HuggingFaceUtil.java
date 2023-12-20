package com.caseystella.llm.embeddings;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.caseystella.llm.DownloadUtil;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class HuggingFaceUtil {

  public static Path findModelFile(Path modelDir) throws FileNotFoundException {
    for(File f: modelDir.toFile().listFiles()) {
      if(f.getName().endsWith(".onnx")) {
        return f.toPath();
      }
    }
    throw new FileNotFoundException(String.format("Could not find a valid onnx file under %s", modelDir.toString()));
  }


}
