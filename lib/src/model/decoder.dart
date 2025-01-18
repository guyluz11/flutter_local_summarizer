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
    Function(int)? onNextWord,
  }) async {
    final OrtSession session = await ModelHelper.loadSession(model.biteList);
    final List<int> currentOutput = [Tokenizer().getPadTokenId()];

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

      // Extract logits and calculate the next token
      final OrtValue? output0 = outputs[0];
      if (output0 == null) {
        printInDebug('Decoder output[0] is null!');
        break;
      }

      final List output0ValueOld = (output0.getValue(
        getOnlyLastElementOfFirstList: true,
      )! as List<List<List>>)
          .first
          .last;

      // Initialize nextTokenId directly and avoid using `map` on the entire list
      final int nextTokenId = _npArgmax(output0ValueOld);
      onNextWord?.call(nextTokenId);
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

    // Flatten the 2D array to return a 1D list
    return currentOutput;
  }

// _npArgmax method definition
  static int _npArgmax(List<dynamic> list) {
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
}
