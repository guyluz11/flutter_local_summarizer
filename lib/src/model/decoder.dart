import 'package:flutter_local_summarizer/src/common_functions.dart';
import 'package:flutter_local_summarizer/src/model/model.dart';
import 'package:flutter_local_summarizer/src/model/model_helper.dart';
import 'package:onnxruntime/onnxruntime.dart';

class Decoder {
  static Future<List<int>?> generateDecode({
    required OrtValueTensor inputOrt,
    required OrtValueTensor attentionMaskOrt,
    required OrtValueTensor encodeOutput,
    required OrtRunOptions runOptions,
    required int maxSummaryLength,
    required int eosTokenId,
    required Model model,
    Function(int)? progress,
  }) async {
    final OrtSession session = await ModelHelper.loadSession(model.biteList);
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
}
