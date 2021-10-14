from ASR_metrics import utils as metrics
# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    # TODO: your code here
    return metrics.calculate_cer(target_text, predicted_text)

def calc_wer(target_text, predicted_text) -> float:
    # TODO: your code here
    wer = metrics.calculate_wer(target_text, predicted_text)
    if wer != 0 and wer < 0.03:
        print(target_text)
        print(predicted_text)
    return metrics.calculate_wer(target_text, predicted_text)
