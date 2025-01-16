import 'package:flutter_local_summarizer/src/common_functions.dart';
import 'package:flutter_local_summarizer/src/model/model.dart';
import 'package:flutter_local_summarizer/src/model/model_helper.dart';
import 'package:flutter_local_summarizer/src/tokenizer.dart';
import 'package:onnxruntime/onnxruntime.dart';

class Decoder {
  static Future<List<int>?> generateDecode({
    required OrtValueTensor attentionMaskOrt,
    required OrtValueTensor encodeOutput,
    required OrtRunOptions runOptions,
    required int maxSummaryLength,
    required int eosTokenId,
    required Model model,
    Function(int)? progress,
  }) async {
    final OrtSession session = await ModelHelper.loadSession(model.biteList);
    List<List<int>> currentOutput = [
      [Tokenizer().getPadTokenId()]
    ];

    printInDebug('Start generateDecode');

    for (int i = 0; i < maxSummaryLength; i++) {
      progress?.call((((i + 1) / maxSummaryLength) * 100).toInt());

      // Prepare inputs for the decoder
      final inputs = {
        'input_ids': OrtValueTensor.createTensorWithDataList(currentOutput),
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

      // Extract logits and calculate the next token
      final OrtValue? output0 = outputs[0];
      if (output0 == null) {
        printInDebug('Decoder output[0] is null!');
        break;
      }
      final List<List<List<double>>> output0Value =
          output0.value! as List<List<List<double>>>;

      // Take the last column along the second axis and find argmax
      final List<List<double>> lastStepLogits =
          output0Value.map((batch) => batch.last).toList();
      final List<int> nextTokenIds = _npArgmax(lastStepLogits);

      // Reshape nextTokenIds to a 2D array with shape (-1, 1)
      final List<List<int>> nextTokenIds2D =
          nextTokenIds.map((id) => [id]).toList();

      // Horizontally stack the new token with the current output
      currentOutput = _hStack(currentOutput, nextTokenIds2D);

      // Release outputs to free resources
      for (final element in outputs) {
        element?.release();
      }

      // Stop if the EOS token is generated
      if (nextTokenIds.contains(eosTokenId)) {
        printInDebug('EOS token encountered. Stopping decoding.');
        break;
      }
    }

    printInDebug('Done generateDecode');
    session.release();

    // Flatten the 2D array to return a 1D list
    return currentOutput.expand((row) => row).toList();
  }

  /// Find argmax for each row (last axis)
  static List<int> _npArgmax(List<List<double>> logits) {
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

  /// Horizontal stack implementation for 2D arrays
  static List<List<int>> _hStack(
    List<List<int>> array1,
    List<List<int>> array2,
  ) {
    if (array1.length != array2.length) {
      throw ArgumentError('Both arrays must have the same number of rows.');
    }

    final List<List<int>> result = [];
    for (int i = 0; i < array1.length; i++) {
      result.add([...array1[i], ...array2[i]]);
    }
    return result;
  }
}
