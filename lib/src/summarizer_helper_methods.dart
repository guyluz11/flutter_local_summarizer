import 'dart:typed_data';

import 'package:flutter_local_summarizer/src/common_functions.dart';
import 'package:flutter_local_summarizer/src/external_links.dart';
import 'package:flutter_local_summarizer/src/model.dart';
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
    int maxSummaryLength = 80,
    Function(int)? progress,
  }) async {
    // final String preprocessTextVar = _preprocessText(inputText);
    final String? summaryOutput = await _summary(
      inputText.toLowerCase(),
      maxSummaryLength: maxSummaryLength,
      progress: progress,
    );

    return summaryOutput;
  }

  Future<OrtSession> _loadSession(Uint8List encoderList) async {
    final sessionOptions = OrtSessionOptions();
    return OrtSession.fromBuffer(encoderList, sessionOptions);
  }

  Future<String?> _summary(
    String text, {
    int maxSummaryLength = 70,
    Function(int)? progress,
  }) async {
    progress?.call(0);
    final Tokenizer tokenizer = Tokenizer();
    final Uint32List encodedUintList = tokenizer.encode(text);

    final List<List<int>> inputList = [encodedUintList.toList()];

    final List<List<int>> attentionMask = _createAttentionMask(inputList);

    final OrtValueTensor inputOrt =
        OrtValueTensor.createTensorWithDataList(inputList);
    final OrtValueTensor attentionMaskOrt =
        OrtValueTensor.createTensorWithDataList(attentionMask);
    final OrtRunOptions runOptions = OrtRunOptions();

    printInDebug('Start');
    final List<List<List<double>>>? outputs = await _generatEncode(
      attentionMaskOrt: attentionMaskOrt,
      inputOrt: inputOrt,
      runOptions: runOptions,
    );

    if (outputs == null) {
      printInDebug('There was error decoding');
      return null;
    }
    final List<List<Float32List>> floatOutputs = _convertToFloat32(outputs);

    final OrtValueTensor encodeOutput =
        OrtValueTensor.createTensorWithDataList(floatOutputs);

    const int eosTokenId =
        1; // Example [EOS] token ID (replace with your actual ID)

    // printInDebug(outputs);
    final List<int>? decodeInts = await generateDecode(
      attentionMaskOrt: attentionMaskOrt,
      inputOrt: inputOrt,
      runOptions: runOptions,
      encodeOutput: encodeOutput,
      maxSummaryLength: maxSummaryLength,
      eosTokenId: eosTokenId,
      progress: progress,
    );

    if (decodeInts == null) {
      printInDebug('There was error decodeInts');
      return null;
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

  Future<List<List<List<double>>>?> _generatEncode({
    required OrtValueTensor inputOrt,
    required OrtValueTensor attentionMaskOrt,
    required OrtRunOptions runOptions,
  }) async {
    List<OrtValue?>? outputs;

    final inputs = {
      'input_ids': inputOrt,
      'attention_mask': attentionMaskOrt,
    };

    printInDebug('Start generatEncode');
    final OrtSession session = await _loadSession(encoderModel.biteList);
    outputs = await session.runAsync(runOptions, inputs);
    printInDebug('Done generatEncode');

    if (outputs == null || outputs.isEmpty) {
      return null;
    }

    final OrtValue? output0 = outputs[0];
    if (output0 == null) {
      return null;
    }
    final List<List<List<double>>> output0Value =
        output0.value! as List<List<List<double>>>;

    for (final element in outputs) {
      element?.release();
    }
    session.release();

    return output0Value;
  }

  Future<List<int>?> generateDecode({
    required OrtValueTensor inputOrt,
    required OrtValueTensor attentionMaskOrt,
    required OrtValueTensor encodeOutput,
    required OrtRunOptions runOptions,
    required int maxSummaryLength,
    required int eosTokenId,
    Function(int)? progress,
  }) async {
    final OrtSession session = await _loadSession(decoderModel.biteList);
    final List<int> currentOutput = [];

    // Start with the initial decoder input
    final List<int> initialDecoderInput = [
      (inputOrt.value as List<List<int>>).first[0],
    ];
    currentOutput.addAll(initialDecoderInput);

    printInDebug('Start generateDecode');

    for (int i = 0; i < maxSummaryLength; i++) {
      progress?.call((((i + 1) / maxSummaryLength) * 100).toInt());

      // Prepare inputs for the decoder
      final inputs = {
        'input_ids': OrtValueTensor.createTensorWithDataList([currentOutput]),
        'encoder_attention_mask': attentionMaskOrt,
        'encoder_hidden_states': encodeOutput,
      };

      // Run the decoder
      final List<OrtValue?>? outputs =
          await session.runAsync(runOptions, inputs);

      if (outputs == null || outputs.isEmpty) {
        printInDebug('Decoder outputs are empty!');
        break;
      }

      // Extract logits and find the next token
      final OrtValue? output0 = outputs[0];
      if (output0 == null) {
        printInDebug('Decoder output[0] is null!');
        break;
      }
      final List<List<List<double>>> output0Value =
          output0.value! as List<List<List<double>>>;
      final List<int> nextTokenIds = _npArgmax(output0Value[0]);
      final int nextTokenId = nextTokenIds.last; // Get the last token ID

      // Append the new token to the current output
      currentOutput.add(nextTokenId);

      // Release outputs to free resources
      for (final element in outputs) {
        element?.release();
      }

      // Stop if the EOS token is generated
      if (nextTokenId == eosTokenId) {
        printInDebug('EOS token encountered. Stopping decoding.');
        break;
      }
    }

    printInDebug('Done generateDecode');
    session.release();

    return currentOutput;
  }

  /// Using axis -1
  List<int> _npArgmax(List<List<double>> logits) {
    final List<int> maxIndices = [];

    for (final List<double> innerList in logits) {
      int maxIndex = 0;
      double maxValue = innerList[0];

      for (int i = 1; i < innerList.length; i++) {
        if (innerList[i] > maxValue) {
          maxValue = innerList[i];
          maxIndex = i;
        }
      }

      maxIndices.add(maxIndex);
    }

    return maxIndices;
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
