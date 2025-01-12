# Flutte Local Summarizer

Flutter package to summarize text

The package is using Falconsai/text_summarization
https://huggingface.co/Falconsai/text_summarization
and run with the ONNX runtime
https://pub.dev/packages/onnxruntime


## How to run

1. Download the files
   https://huggingface.co/Falconsai/text_summarization/blob/main/onnx/encoder_model.onnx
   https://huggingface.co/Falconsai/text_summarization/blob/main/onnx/decoder_model.onnx
   and set the names to
   flassco_decoder_model.onnx
   flassco_encoder_model.onnx
2. Insert the files under assets/models folder
3. Add the path assets/models/ in pubspec.yaml
    flutter:
       assets:
        - assets/models/
4. Call SummarizerHelperMethods.init(); in the main function
5. Call SummarizerHelperMethods().flasscoSummarize(text) with the text you want
6. Done, So simple there is no more to do :)
