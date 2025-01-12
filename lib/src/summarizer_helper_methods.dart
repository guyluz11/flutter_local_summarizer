import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:langchain_tiktoken/langchain_tiktoken.dart';
import 'package:onnxruntime/onnxruntime.dart';

class SummarizerHelperMethods {
  static void init() => OrtEnv.instance.init();

  Future<String?> flasscoSummarize(
    String inputText, {
    int maxSummaryLength = 80,
    Function(int)? progress,
  }) async {
    // final String preprocessTextVar = _preprocessText(inputText);
    final String? summeryOutput = await _summery(
      inputText.toLowerCase(),
      maxSummaryLength: maxSummaryLength,
      progress: progress,
    );

    return summeryOutput;
  }

  Future<OrtSession> _loadDecoderSession() async {
    final sessionOptions = OrtSessionOptions();
    const assetFileName = 'assets/models/flassco_decoder_model.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    return OrtSession.fromBuffer(bytes, sessionOptions);
  }

  Future<OrtSession> _loadEncoderSession() async {
    final sessionOptions = OrtSessionOptions();
    const assetFileName = 'assets/models/flassco_encoder_model.onnx';
    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    return OrtSession.fromBuffer(bytes, sessionOptions);
  }

  Future<String?> _summery(
    String text, {
    int maxSummaryLength = 70,
    Function(int)? progress,
  }) async {
    progress?.call(0);
    final Tiktoken tiktoken = encodingForModel("t5");
    final Uint32List encodedUintList = tiktoken.encode(text);

    final List<List<int>> inputList = [encodedUintList.toList()];

    final List<List<int>> attentionMask = _createAttentionMask(inputList);

    final OrtValueTensor inputOrt =
        OrtValueTensor.createTensorWithDataList(inputList);
    final OrtValueTensor attentionMaskOrt =
        OrtValueTensor.createTensorWithDataList(attentionMask);
    final OrtRunOptions runOptions = OrtRunOptions();
    print('Start');
    final List<List<List<double>>>? outputs = await _generatEncode(
      attentionMaskOrt: attentionMaskOrt,
      inputOrt: inputOrt,
      runOptions: runOptions,
    );

    if (outputs == null) {
      print('There was error decoding');
      return null;
    }
    final List<List<Float32List>> floatOutputs = _convertToFloat32(outputs);

    final OrtValueTensor encodeOutput =
        OrtValueTensor.createTensorWithDataList(floatOutputs);

    final int eosTokenId =
        1; // Example [EOS] token ID (replace with your actual ID)

    // print(outputs);
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
      print('There was error decodeInts');
      return null;
    }

    final String decodeString = tiktoken.decode(decodeInts);

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
        .map((outerList) => outerList
            .map((innerList) => Float32List.fromList(innerList))
            .toList())
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

    print('Start generatEncode');
    final OrtSession session = await _loadEncoderSession();
    outputs = await session.runAsync(runOptions, inputs);
    print('Done generatEncode');

    if (outputs == null || outputs.isEmpty) {
      return null;
    }

    final OrtValue? output0 = outputs[0];
    if (output0 == null) {
      return null;
    }
    final List<List<List<double>>> output0Value =
        output0.value! as List<List<List<double>>>;

    outputs.forEach((element) {
      element?.release();
    });
    session.release();

    return output0Value;
  }

  Future<List<int>?> generateDecode({
    required OrtValueTensor inputOrt,
    required OrtValueTensor attentionMaskOrt,
    required OrtValueTensor encodeOutput,
    required OrtRunOptions runOptions,
    required int maxSummaryLength, // Maximum summary length
    required int eosTokenId, // End-of-sequence token ID
    Function(int)? progress,
  }) async {
    final OrtSession session = await _loadDecoderSession();
    final List<int> currentOutput = []; // Stores the generated token IDs

    // Start with the initial decoder input (e.g., [BOS] token ID)
    final List<int> initialDecoderInput = [
      inputOrt.value.first[0] as int
    ]; // Assuming the first token
    currentOutput.addAll(initialDecoderInput);

    print('Start generateDecode');

    // Iterate up to the maximum summary length
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
        print('Decoder outputs are empty!');
        break;
      }

      // Extract logits and find the next token
      final OrtValue? output0 = outputs[0];
      if (output0 == null) {
        print('Decoder output[0] is null!');
        break;
      }
      final List<List<List<double>>> output0Value =
          output0.value! as List<List<List<double>>>;
      final List<int> nextTokenIds = _npArgmax(output0Value[0]);
      final int nextTokenId = nextTokenIds.last; // Get the last token ID

      // Append the new token to the current output
      currentOutput.add(nextTokenId);

      // Release outputs to free resources
      outputs.forEach((element) {
        element?.release();
      });

      // Stop if the [EOS] token is generated
      if (nextTokenId == eosTokenId) {
        print('EOS token encountered. Stopping decoding.');
        break;
      }
    }
    print('Done generateDecode');

    session.release();

    return currentOutput;
  }

  /// Using axis -1
  List<int> _npArgmax(List<List<double>> logits) {
    final List<int> maxIndices = [];

    for (List<double> innerList in logits) {
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

  String _preprocessText(String text) {
    // Remove consecutive punctuation (e.g., `...,`)
    text = text.replaceAll(RegExp(r'[,.]{2,}'), ' ');

    // Remove extra spaces
    text = text.replaceAll(RegExp(r'\s+'), ' ').trim();

    // Remove non-ASCII characters
    text = text.replaceAll(RegExp(r'[^\x00-\x7F]+'), ' ');

    // Lowercase the text (optional)
    text = text.toLowerCase();

    return text;
  }
}
