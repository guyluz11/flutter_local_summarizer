class ExternalLinks {
  static String flasscoEncoderModelLocation =
      'assets/models/flassco_encoder_model.onnx';

  static Uri flasscoEncoderModelUrl = Uri.parse(
    'https://huggingface.co/Falconsai/text_summarization/resolve/main/onnx/encoder_model.onnx?download=true',
  );

  static String flasscoDecoderModelLocation =
      'assets/models/flassco_decoder_model.onnx';
  static Uri flasscoDecoderModelUrl = Uri.parse(
    'https://huggingface.co/Falconsai/text_summarization/resolve/main/onnx/decoder_model.onnx?download=true',
  );
}
