package com.caseystella.llm;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DownloadUtil {
  private static final Logger logger = LoggerFactory.getLogger(DownloadUtil.class);

  /**
   * Joins two parts of a URL without doubling the '/'.
   *
   * @param part1 The first part of the URL.
   * @param part2 The second part of the URL.
   * @return The joined URL.
   */
  public static String joinURLParts(String part1, String part2) {
    if (part1.endsWith("/") && part2.startsWith("/")) {
      // Both parts end and start with a slash, remove one
      return part1 + part2.substring(1);
    } else if (!part1.endsWith("/") && !part2.startsWith("/")) {
      // Neither part ends/starts with a slash, add one
      return part1 + "/" + part2;
    } else {
      // Only one of the parts has a slash at the junction, no change needed
      return part1 + part2;
    }
  }

  public static File downloadFile(String fileURL, String targetDirectory) throws IOException {

    URL url = new URL(fileURL);
    HttpURLConnection connection = (HttpURLConnection) url.openConnection();
    int fileSize = connection.getContentLength();
    String fileName = url.getFile().substring(url.getFile().lastIndexOf('/') + 1);
    File targetFile = new File(targetDirectory, fileName);
    if (targetFile.exists()) {
      logger.info("File already exists, skipping...");
      return targetFile;
    }
    logger.info("Downloading {} to {}", fileURL, targetDirectory);
    File targetDirectoryFile = new File(targetDirectory);
    if (!targetDirectoryFile.exists()) {
      boolean mkdirs = targetDirectoryFile.mkdirs();
      if (!mkdirs) {
        throw new IllegalStateException(
            String.format("Unable to create directory %s", targetDirectory));
      }
    }
    try (BufferedInputStream in = new BufferedInputStream(connection.getInputStream());
        FileOutputStream fileOutputStream = new FileOutputStream(targetFile)) {
      byte dataBuffer[] = new byte[1024];
      int bytesRead;
      long totalBytesRead = 0;
      int previousProgress = 0;

      while ((bytesRead = in.read(dataBuffer, 0, 1024)) != -1) {
        fileOutputStream.write(dataBuffer, 0, bytesRead);
        totalBytesRead += bytesRead;
        int currentProgress = (int) ((totalBytesRead * 100L) / fileSize);

        if (currentProgress > previousProgress) {
          logger.info("Download progress: {}%", currentProgress);
          previousProgress = currentProgress;
        }
      }

      logger.info("File downloaded successfully to {}", targetDirectory);
    } finally {
      connection.disconnect();
    }
    return targetFile;
  }

  public static File[] downloadFiles(String targetDirectory, String... fileURLs)
      throws IOException {
    File[] ret = new File[fileURLs.length];
    for (int i = 0; i < fileURLs.length; ++i) {
      ret[i] = downloadFile(fileURLs[i], targetDirectory);
    }
    return ret;
  }
}
