# Define pattern arrays
$tacolumns_SpecialPreprocessed = @(
    'volume_em', 'volume_sma_em', 'volume_nvi', 'volatility_kcp', 'trend_dpo',
    'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_adx', 'trend_adx_pos',
    'trend_adx_neg', 'momentum_ppo', 'momentum_ppo_signal', 'momentum_ppo_hist',
    'volume_obv', 'volume_fi', 'volume_vpt', 'trend_trix'
)

$tacolumns_DivideByHundreds = @(
    'volume_mfi', 'volatility_ui', 'trend_mass_index', 'trend_aroon_up',
    'trend_aroon_down', 'trend_aroon_ind', 'momentum_tsi', 'momentum_uo',
    'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'momentum_roc',
    'trend_stc', 'trend_cci', 'momentum_pvo_signal', 'momentum_rsi',
    'momentum_pvo', 'momentum_pvo_hist'
)

$tacolumns_AsIs = @(
    'volume_cmf', 'volatility_bbhi', 'trend_psar_down_indicator', 'volatility_kcli',
    'volatility_bbli', 'trend_psar_up_indicator', 'volatility_kchi', 'momentum_stoch_rsi',
    'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'volatility_dcp',
    'trend_vortex_ind_neg', 'trend_vortex_ind_pos', 'trend_vortex_ind_diff',
    'volatility_bbp'
)

$tacolumns_ScaledSpecial = @(
    'Volume', 'volume_adi'
)

$tacolumns_ScaledToClose = @(
    'Open', 'High', 'Low', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
    'trend_sma_slow', 'volatility_atr', 'volatility_bbl', 'volatility_bbm',
    'volatility_bbh', 'volatility_bbw', 'volatility_kcc', 'momentum_ao',
    'trend_psar_down', 'trend_psar_up', 'trend_sma_fast', 'volatility_kch',
    'volatility_kcl', 'volume_vwap', 'trend_ichimoku_base', 'trend_ichimoku_conv',
    'trend_ichimoku_a', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',
    'trend_ichimoku_b', 'trend_ema_slow', 'momentum_kama', 'trend_ema_fast',
    'volatility_dcm', 'volatility_dcl', 'volatility_dch'
)

# Combine all patterns into one array
$allPatterns = $tacolumns_SpecialPreprocessed + $tacolumns_DivideByHundreds + $tacolumns_AsIs + $tacolumns_ScaledSpecial + $tacolumns_ScaledToClose

# Define the search path (current directory) and file filter
$searchPath = Get-Location
$filter = "output_*"

# For each pattern, count and output the number of occurrences
foreach ($pattern in $allPatterns) {
    $matches = Get-ChildItem -Path $searchPath -Filter $filter -File -Recurse | Select-String -Pattern $pattern
    $count = if ($matches) { $matches.Count } else { 0 }
    Write-Output "$($pattern): $($count)"
}
