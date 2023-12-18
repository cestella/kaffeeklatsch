package com.caseystella.llm.nlp;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;

public class NLPUtil {

  public static SentenceModel loadModel(Path nlpModel) throws IOException {

    try (InputStream modelIn = new FileInputStream(nlpModel.toFile())) {
      return new SentenceModel(modelIn);
    }
  }

  public static String[] splitBySentence(String text, SentenceModel model) {
    SentenceDetectorME sentenceDetector = new SentenceDetectorME(model);

    return sentenceDetector.sentDetect(text);
  }
}
