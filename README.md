# Flutter Local Summarizer

Flutter package to summarize text

The package is using Falconsai/text_summarization
https://huggingface.co/Falconsai/text_summarization
and run with the ONNX runtime
https://pub.dev/packages/onnxruntime


## How to run

1. Call `await SummarizerHelperMethods.init();` and wait till finish.
2. Run `SummarizerHelperMethods().flasscoSummarize(text)` with the text you desire to summarize. you can also pass `progress:` and pass a progress callback.
3. Done, So simple there is no more to do 😁👍
