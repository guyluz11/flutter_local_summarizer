import 'package:flutter_local_summarizer/src/common_functions.dart';
import 'package:flutter_local_summarizer/src/model/model.dart';
import 'package:flutter_local_summarizer/src/model/model_helper.dart';
import 'package:onnxruntime/onnxruntime.dart';

class Encoder {
  static Future<List<List<List<double>>>?> generateEncode({
    required OrtValueTensor inputOrt,
    required OrtValueTensor attentionMaskOrt,
    required OrtRunOptions runOptions,
    required Model model,
  }) async {
    List<OrtValue?>? outputs;

    final inputs = {
      'input_ids': inputOrt,
      'attention_mask': attentionMaskOrt,
    };

    printInDebug('Start generatEncode');
    final OrtSession session = await ModelHelper.loadSession(model.biteList);
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
        output0.getValue()! as List<List<List<double>>>;

    for (final element in outputs) {
      element?.release();
    }
    session.release();

    return output0Value;
  }
}
