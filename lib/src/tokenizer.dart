import 'dart:typed_data';

import 'package:langchain_tiktoken/langchain_tiktoken.dart';

class Tokenizer {
  Tokenizer() {
    tiktoken = encodingForModel('t5');
  }

  late Tiktoken tiktoken;

  Uint32List encode(String text) => tiktoken.encode(text);

  String decode(List<int> decodeInts) => tiktoken.decode(decodeInts);

  /// [PAD] token ID
  int getPadTokenId() => 0;

  /// [EOS] token ID
  int getEosTokenId() => 1;
}
