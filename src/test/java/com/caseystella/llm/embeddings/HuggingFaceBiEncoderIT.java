package com.caseystella.llm.embeddings;

import ai.djl.util.PairList;
import ai.djl.util.StringPair;
import ai.onnxruntime.OrtEnvironment;
import com.caseystella.llm.nlp.NLPUtil;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class HuggingFaceBiEncoderIT {

  private OrtEnvironment environment;
  private Path embeddingModelHome;
  private Path nlpModelHome;

  @BeforeEach
  public void setup() {
    embeddingModelHome = Path.of(System.getProperty("embedding.model.home"));
    nlpModelHome = Path.of(System.getProperty("nlp.model.home"));
    environment = OrtEnvironment.getEnvironment();
  }

  @AfterEach
  public void teardown() {
    if (environment != null) {
      environment.close();
    }
  }

  @Test
  public void testBiEncoder() throws Exception {
    IBiEncoder encoder = BiEncoders.ALL_MPNET_BASE_V2.create(embeddingModelHome);
    String[] sentences = new String[] {"Hello world", "Greetings planet"};
    Iterator<float[]> embeddingsIt = encoder.embed(sentences, Optional.of(environment)).iterator();
    float[] embedding1 = embeddingsIt.next();
    float[] embedding2 = embeddingsIt.next();
    Assertions.assertFalse(embeddingsIt.hasNext());
    double cosSim = EncodingUtil.cosineSimilarity(embedding1, embedding2);
    Assertions.assertTrue(cosSim >= 0.7 && cosSim < 1.0);
  }

  @Test
  public void testCrossEncoder_basic() throws Exception {
    HuggingFaceCrossEncoder encoder =
        CrossEncoders.MS_MARCO_MiniLM_L6_v2.create(embeddingModelHome);
    PairList<String, String> input =
        new PairList<>() {
          {
            add(
                "How many people live in Berlin?",
                "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82"
                    + " square kilometers.");
            add("How many people live in Berlin?", "Berlin is well known for its museums.");
          }
        };
    float[] res = encoder.embed(input, Optional.of(environment));
    Assertions.assertTrue(Math.abs(res[0] - 8.607141f) < 0.1f);
    Assertions.assertTrue(Math.abs(res[1] - -4.32008f) < 0.1f);
  }

  private static List<String> groupSentences(List<String> sentences, int groupSize) {
    List<String> finalParagraphs = new ArrayList<>();
    List<String> buff = new ArrayList<>();
    for (int i = 0; i < sentences.size(); ++i) {
      if (i > 0 && i % groupSize == 0) {
        finalParagraphs.add(String.join(" ", buff));
        buff = new ArrayList<>();
      }
      buff.add(sentences.get(i));
    }
    if (!buff.isEmpty()) {
      finalParagraphs.add(String.join(" ", buff));
    }
    return finalParagraphs;
  }

  @Test
  public void testCrossEncoder() throws Exception {
    HuggingFaceCrossEncoder encoder =
        CrossEncoders.MS_MARCO_MiniLM_L6_v2.create(embeddingModelHome);
    String doc =
        "Europe is a continent located entirely in the Northern Hemisphere and mostly in the"
            + " Eastern Hemisphere. It comprises the westernmost part of Eurasia and is bordered by"
            + " the Arctic Ocean to the north, the Atlantic Ocean to the west, the Mediterranean"
            + " Sea to the south, and Asia to the east. Europe is commonly considered to be"
            + " separated from Asia by the watershed of the Ural Mountains, the Ural River, the"
            + " Caspian Sea, the Greater Caucasus, the Black Sea, and the waterways of the Turkish"
            + " Straits. Although some of this border is over land, Europe is generally accorded"
            + " the status of a full continent because of its great physical size and the weight of"
            + " history and tradition.\n"
            + "\n"
            + "Europe covers about 10,180,000 square kilometres (3,930,000 sq mi), or 2% of the"
            + " Earth's surface (6.8% of land area), making it the second smallest continent."
            + " Politically, Europe is divided into about fifty sovereign states, of which Russia"
            + " is the largest and most populous, spanning 39% of the continent and comprising 15%"
            + " of its population. Europe had a total population of about 741 million (about 11% of"
            + " the world population) as of 2018. The European climate is largely affected by warm"
            + " Atlantic currents that temper winters and summers on much of the continent, even at"
            + " latitudes along which the climate in Asia and North America is severe. Further from"
            + " the sea, seasonal differences are more noticeable than close to the coast.\n"
            + "\n"
            + "European culture is the root of Western civilization, which traces its lineage back"
            + " to ancient Greece and ancient Rome. The fall of the Western Roman Empire in 476 AD"
            + " and the subsequent Migration Period marked the end of Europe's ancient history and"
            + " the beginning of the Middle Ages. Renaissance humanism, exploration, art and"
            + " science led to the modern era. Since the Age of Discovery, started by Portugal and"
            + " Spain, Europe played a predominant role in global affairs. Between the 16th and"
            + " 20th centuries, European powers colonized at various times the Americas, almost all"
            + " of Africa and Oceania, and the majority of Asia.\n"
            + "\n"
            + "The Age of Enlightenment, the subsequent French Revolution and the Napoleonic Wars"
            + " shaped the continent culturally, politically and economically from the end of the"
            + " 17th century until the first half of the 19th century. The Industrial Revolution,"
            + " which began in Great Britain at the end of the 18th century, gave rise to radical"
            + " economic, cultural and social change in Western Europe and eventually the wider"
            + " world. Both world wars took place for the most part in Europe, contributing to a"
            + " decline in Western European dominance in world affairs by the mid-20th century as"
            + " the Soviet Union and the United States took prominence. During the Cold War, Europe"
            + " was divided along the Iron Curtain between NATO in the West and the Warsaw Pact in"
            + " the East, until the revolutions of 1989 and fall of the Berlin Wall.\n"
            + "\n"
            + "In 1949, the Council of Europe was founded with the idea of unifying Europe to"
            + " achieve common goals. Further European integration by some states led to the"
            + " formation of the European Union (EU), a separate political entity that lies between"
            + " a confederation and a federation. The EU originated in Western Europe but has been"
            + " expanding eastward since the fall of the Soviet Union in 1991. The currency of most"
            + " countries of the European Union, the euro, is the most commonly used among"
            + " Europeans; and the EU's Schengen Area abolishes border and immigration controls"
            + " between most of its member states. There exists a political movement favoring the"
            + " evolution of the European Union into a single federation encompassing much of the"
            + " continent.\n"
            + "\n"
            + "In classical Greek mythology, Europa (Ancient Greek: Εὐρώπη, Eurṓpē) was a"
            + " Phoenician princess. One view is that her name derives from the ancient Greek"
            + " elements εὐρύς (eurús), \"wide, broad\" and ὤψ (ōps, gen. ὠπός, ōpós) \"eye, face,"
            + " countenance\", hence their composite Eurṓpē would mean \"wide-gazing\" or \"broad"
            + " of aspect\". Broad has been an epithet of Earth herself in the reconstructed"
            + " Proto-Indo-European religion and the poetry devoted to it. An alternative view is"
            + " that of R.S.P. Beekes who has argued in favor of a Pre-Indo-European origin for the"
            + " name, explaining that a derivation from ancient Greek eurus would yield a different"
            + " toponym than Europa. Beekes has located toponyms related to that of Europa in the"
            + " territory of ancient Greece and localities like that of Europos in ancient"
            + " Macedonia.\n"
            + "\n"
            + "There have been attempts to connect Eurṓpē to a Semitic term for \"west\", this"
            + " being either Akkadian erebu meaning \"to go down, set\" (said of the sun) or"
            + " Phoenician 'ereb \"evening, west\", which is at the origin of Arabic Maghreb and"
            + " Hebrew ma'arav. Michael A. Barry finds the mention of the word Ereb on an Assyrian"
            + " stele with the meaning of \"night, [the country of] sunset\", in opposition to Asu"
            + " \"[the country of] sunrise\", i.e. Asia. The same naming motive according to"
            + " \"cartographic convention\" appears in Greek Ἀνατολή (Anatolḗ \"[sun] rise\","
            + " \"east\", hence Anatolia). Martin Litchfield West stated that \"phonologically, the"
            + " match between Europa's name and any form of the Semitic word is very poor\", while"
            + " Beekes considers a connection to Semitic languages improbable. Next to these"
            + " hypotheses there is also a Proto-Indo-European root *h1regʷos, meaning"
            + " \"darkness\", which also produced Greek Erebus.\n"
            + "\n"
            + "Most major world languages use words derived from Eurṓpē or Europa to refer to the"
            + " continent. Chinese, for example, uses the word Ōuzhōu (歐洲/欧洲), which is an"
            + " abbreviation of the transliterated name Ōuluóbā zhōu (歐羅巴洲) (zhōu means"
            + " \"continent\"); a similar Chinese-derived term Ōshū (欧州) is also sometimes used in"
            + " Japanese such as in the Japanese name of the European Union, Ōshū Rengō (欧州連合),"
            + " despite the katakana Yōroppa (ヨーロッパ) being more commonly used. In some Turkic"
            + " languages, the originally Persian name Frangistan (\"land of the Franks\") is used"
            + " casually in referring to much of Europe, besides official names such as Avrupa or"
            + " Evropa.";
    String query = "How large is Europe?";
    String[] paragraphs = doc.split("\n\n");
    var model =
        NLPUtil.loadModel(
            Path.of(nlpModelHome.toString(), "opennlp-en-ud-ewt-sentence-1.0-1.9.3.bin"));
    List<String> sentences = new ArrayList<>();
    for (String paragraph : paragraphs) {
      for (String s : NLPUtil.splitBySentence(paragraph, model)) {
        sentences.add(s.trim());
      }
    }
    List<String> finalParagraphs = groupSentences(sentences, 3);
    PairList<String, String> input = new PairList<>();
    for (String p : finalParagraphs) {
      input.add(new StringPair(query, p));
    }
    float[] logits = encoder.embed(input, Optional.of(environment));
    int maxIndex = -1;
    float maxLogit = 0;
    for (int i = 0; i < logits.length; ++i) {
      if (maxIndex == -1 || logits[i] > maxLogit) {
        maxLogit = logits[i];
        maxIndex = i;
      }
    }
    Assertions.assertEquals(maxIndex, 1);
  }
}
