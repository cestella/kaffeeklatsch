package com.caseystella.llm.nlp;

import com.caseystella.llm.DownloadUtil;
import com.caseystella.llm.ModelHome;
import java.io.IOException;
import java.nio.file.Path;
import opennlp.tools.sentdetect.SentenceModel;

public enum SentenceModels {
  ENGLISH(
      "opennlp-en-sentence_detector.bin",
      "https://dlcdn.apache.org/opennlp/models/ud-models-1.0/opennlp-en-ud-ewt-sentence-1.0-1.9.3.bin");
  String name;
  String url;

  SentenceModels(String name, String url) {
    this.name = name;
    this.url = url;
  }

  public SentenceModel create() throws IOException {
    return create(ModelHome.NLP.get());
  }

  public SentenceModel create(Path nlpModelPath) throws IOException {
    return NLPUtil.loadModel(DownloadUtil.downloadFile(name, nlpModelPath.toString()).toPath());
  }
}
