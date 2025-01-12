import 'package:flutter/foundation.dart';

void printInDebug(String value) {
  if (kDebugMode) {
    print(value);
  }
}
