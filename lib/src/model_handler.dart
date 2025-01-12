import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_local_summarizer/src/common_functions.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

class ModelHandler {
  static Future<Uint8List> loadModelBytes(
    Uri modelUrl,
    String localFileName,
  ) async {
    // Get the directory to save the file
    final directory = await getApplicationDocumentsDirectory();
    final filePath = '${directory.path}/$localFileName';
    final file = File(filePath);

    // Ensure the directory exists
    final dir = file.parent;
    if (!await dir.exists()) {
      printInDebug('Creating missing directories: ${dir.path}');
      await dir.create(recursive: true);
    }

    // Check if the file already exists locally
    if (await file.exists()) {
      printInDebug('Model loaded from local storage.');
      return await file.readAsBytes();
    }

    // Download the model from the given URL
    printInDebug('Downloading model from $modelUrl...');
    final response = await http.get(modelUrl);
    if (response.statusCode == 200) {
      // Save the file locally
      await file.writeAsBytes(response.bodyBytes);
      printInDebug('Model downloaded and saved to $filePath');
      return response.bodyBytes;
    } else {
      throw Exception('Failed to download model: ${response.statusCode}');
    }
  }
}
