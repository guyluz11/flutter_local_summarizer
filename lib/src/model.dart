import 'package:flutter/foundation.dart';
import 'package:flutter_local_summarizer/src/model_handler.dart';

class Model {
  Model({required this.url, required this.saveLocation});

  Uri url;
  String saveLocation;
  late Uint8List biteList;

  Future innit() async {
    biteList = await ModelHandler.loadModelBytes(url, saveLocation);
  }
}
