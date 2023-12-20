package com.caseystella.llm;

import java.io.FileNotFoundException;
import java.nio.file.Path;

public enum ModelHome {
  EMBEDDINGS("embeddings"),
  NLP("nlp"),
  LLM("llm");
  String type;

  ModelHome(String type) {
    this.type = type;
  }

  public Path get(Path modelHome) throws FileNotFoundException {
    if (!modelHome.toFile().exists()) {
      throw new FileNotFoundException(modelHome + " does not exist!");
    } else {
      return Path.of(modelHome.toString(), type);
    }
  }

  public Path get() throws FileNotFoundException {
    String modelHome = System.getProperty("model.home");
    if (modelHome == null) {
      return get(Path.of("models"));
    } else {
      return get(Path.of(modelHome));
    }
  }
}
