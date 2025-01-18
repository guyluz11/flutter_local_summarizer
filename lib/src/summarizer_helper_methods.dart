import 'dart:typed_data';

import 'package:flutter_local_summarizer/src/common_functions.dart';
import 'package:flutter_local_summarizer/src/external_links.dart';
import 'package:flutter_local_summarizer/src/model/decoder.dart';
import 'package:flutter_local_summarizer/src/model/encoder.dart';
import 'package:flutter_local_summarizer/src/model/model.dart';
import 'package:flutter_local_summarizer/src/tokenizer.dart';
import 'package:onnxruntime/onnxruntime.dart';

class SummarizerHelperMethods {
  late Model decoderModel;
  late Model encoderModel;

  Future init() async {
    OrtEnv.instance.init();
    decoderModel = Model(
      url: ExternalLinks.flasscoDecoderModelUrl,
      saveLocation: ExternalLinks.flasscoDecoderModelLocation,
    );
    encoderModel = Model(
      url: ExternalLinks.flasscoEncoderModelUrl,
      saveLocation: ExternalLinks.flasscoEncoderModelLocation,
    );
    await Future.wait([
      decoderModel.innit(),
      encoderModel.innit(),
    ]);
  }

  Future<String?> flasscoSummarize(
    String inputText, {
    int maxSummaryLength = 300,
    Function(int)? progress,
    Function(String)? onNextWord,
  }) async {
    // final String preprocessTextVar = _preprocessText(inputText);
    final String preprocessTextVar = inputText;
    final String? summaryOutput = await _summary(
      preprocessTextVar,
      maxSummaryLength: maxSummaryLength,
      progress: progress,
      onNextWord: onNextWord,
    );

    return summaryOutput;
  }

  Future<String?> _summary(
    String text, {
    int maxSummaryLength = 300,
    Function(int)? progress,
    Function(String)? onNextWord,
  }) async {
    progress?.call(0);
    final Tokenizer tokenizer = Tokenizer();
    final Uint32List encodedUintList = tokenizer.encode(text);

    final List<List<int>> inputList = [
      [tokenizer.getPadTokenId()] +
          encodedUintList.toList() +
          [tokenizer.getEosTokenId()],
    ];

    final List<List<int>> attentionMask = _createAttentionMask(inputList);

    final OrtValueTensor inputOrt =
        OrtValueTensor.createTensorWithDataList(inputList);
    final OrtValueTensor attentionMaskOrt =
        OrtValueTensor.createTensorWithDataList(attentionMask);
    final OrtRunOptions runOptions = OrtRunOptions();

    printInDebug('Start');
    final List<List<List<double>>>? outputs = await Encoder.generateEncode(
      attentionMaskOrt: attentionMaskOrt,
      inputOrt: inputOrt,
      runOptions: runOptions,
      model: encoderModel,
    );

    if (outputs == null) {
      printInDebug('There was error decoding');
      return null;
    }

    final List<List<Float32List>> floatOutputs = _convertToFloat32(outputs);

    final OrtValueTensor encodeOutput =
        OrtValueTensor.createTensorWithDataList(floatOutputs);

    // printInDebug(outputs);
    List<int>? decodeInts = await Decoder.generateDecode(
      attentionMaskOrt: attentionMaskOrt,
      runOptions: runOptions,
      encodeOutput: encodeOutput,
      maxSummaryLength: maxSummaryLength,
      eosTokenId: tokenizer.getEosTokenId(),
      progress: progress,
      model: decoderModel,
      onNextWord: (int word) {
        onNextWord?.call(tokenizer.decode([word]));
      },
    );

    if (decodeInts == null) {
      printInDebug('There was error decodeInts');
      return null;
    }

    // Remove start token
    if (decodeInts[0] == tokenizer.getPadTokenId()) {
      decodeInts = decodeInts.sublist(1);
    }

    // Remove start token
    final int decodeIntsLength = decodeInts.length;
    if (decodeInts[decodeIntsLength - 1] == tokenizer.getEosTokenId()) {
      decodeInts = decodeInts.sublist(0, decodeIntsLength - 1);
    }

    final String decodeString = tokenizer.decode(decodeInts);

    inputOrt.release();
    attentionMaskOrt.release();
    runOptions.release();
    encodeOutput.release();

    progress?.call(100);

    return decodeString;
  }

  /// Helper function to flatten and convert nested List<List<List<double>>> to Float32List
  List<List<Float32List>> _convertToFloat32(List<List<List<double>>> data) {
    // Convert the nested list to a nested Float32List
    return data
        .map(
          (outerList) => outerList
              .map((innerList) => Float32List.fromList(innerList))
              .toList(),
        )
        .toList();
  }

  List<List<int>> _createAttentionMask(List<List<int>> inputList) {
    // Iterate over each inner list to create the attention mask
    return inputList.map((sequence) {
      return sequence.map((token) => token != 0 ? 1 : 0).toList();
    }).toList();
  }

  // String _preprocessText(String text) {
  //   // Remove consecutive punctuation (e.g., `...,`)
  //   String tempText = text.replaceAll(RegExp('[,.]{2,}'), ' ');
  //
  //   // Remove extra spaces
  //   tempText = tempText.replaceAll(RegExp(r'\s+'), ' ').trim();
  //
  //   // Remove non-ASCII characters
  //   tempText = tempText.replaceAll(RegExp(r'[^\x00-\x7F]+'), ' ');
  //
  //   // Lowercase the text (optional)
  //   tempText = tempText.toLowerCase();
  //
  //   return tempText;
  // }
}
