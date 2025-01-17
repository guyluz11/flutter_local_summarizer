import 'package:flutter/material.dart';
import 'package:flutter_local_summarizer/flutter_local_summarizer.dart';
import 'package:summarizer_example/helpers/texts.dart';

late SummarizerHelperMethods summarizerHelperMethods;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  summarizerHelperMethods = SummarizerHelperMethods();
  await summarizerHelperMethods.init();
  runApp(
    MaterialApp(
      home: TextSummarization(),
    ),
  );
}

class TextSummarization extends StatefulWidget {
  @override
  _TextSummarizationState createState() => _TextSummarizationState();
}

class _TextSummarizationState extends State<TextSummarization> {
  String summary = '';
  int progressVar = 0;

  Future getSummary({String? text}) async {
    setState(() {
      summary = 'Generating summary...';
    });

    final String? summarizedText =
        await summarizerHelperMethods.flasscoSummarize(
      text ?? getLongText,
      progress: progress,
      maxSummaryLength: 100,
      onNextWord: onNextWord,
    );

    setState(() {
      summary = summarizedText ?? 'Error summarizing';
    });
  }

  void onNextWord(String word) {
    setState(() {
      summary += word;
    });
  }

  void progress(int progressTemp) {
    setState(() {
      progressVar = progressTemp;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Text Summarization with ONNX'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextButton(
              onPressed: getSummary,
              style: ButtonStyle(
                backgroundColor: WidgetStateProperty.all<Color>(Colors.grey),
              ),
              child: const Text(
                'Press to impress',
                style: TextStyle(fontSize: 15),
              ),
            ),
            const SizedBox(height: 20),
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Summary:', style: TextStyle(fontSize: 20)),
                    const SizedBox(height: 10, width: double.infinity),
                    Text('Progress: $progressVar'),
                    const SizedBox(height: 10),
                    Text(summary),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
