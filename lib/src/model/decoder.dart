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
    List<int> currentOutput = [Tokenizer().getPadTokenId()];

    printInDebug('Start generateDecode');

    // Create a Stopwatch instance
    Stopwatch stopwatch = Stopwatch();

    // Start the stopwatch
    stopwatch.start();
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

      // Extract logits and calculate the next token
      final OrtValue? output0 = outputs[0];
      if (output0 == null) {
        printInDebug('Decoder output[0] is null!');
        break;
      }

      final List<double> output0ValueOlde =
          (output0.value! as List<List<List<double>>>).first.last;

      // Initialize nextTokenId directly and avoid using `map` on the entire list
      int nextTokenId = output0ValueOlde.first.toInt();
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

    // Stop the stopwatch
    stopwatch.stop();

    // Get the elapsed time in milliseconds
    print('Time taken: ${stopwatch.elapsedMilliseconds} milliseconds');
    printInDebug('Done generateDecode');
    session.release();

    // Flatten the 2D array to return a 1D list
    return currentOutput;
  }

// _npArgmax method definition
  static int _npArgmax(List<double> list) {
    int maxIndex = 0;
    double maxValue = list[0] as double;
    for (int i = 1; i < list.length; i++) {
      final double listI = list[i] as double;
      if (listI > maxValue) {
        maxValue = listI;
        maxIndex = i;
      }
    }
    return maxIndex;
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
