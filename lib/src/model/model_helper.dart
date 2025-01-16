import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class ModelHelper {
  static Future<OrtSession> loadSession(Uint8List encoderList) async {
    final sessionOptions = OrtSessionOptions();
    return OrtSession.fromBuffer(encoderList, sessionOptions);
  }
}
