$path = 'c:\Users\ander\OneDrive\Skrivebord\6. semester\Bachelor Projekt\Kode\min_kode\quadcopter_bach_piml\test_main.ipynb'
$json = Get-Content -Raw $path | ConvertFrom-Json
$cell = $json.cells | Where-Object { $_.id -eq '#VSC-08029779' -and $_.cell_type -eq 'code' } | Select-Object -First 1
if (-not $cell) { throw 'Target cell not found.' }
$src = [System.Collections.ArrayList]@($cell.source)
$start = $src.IndexOf('def build_multistep_sequences_from_ordered_data(')
$end = $src.IndexOf('    return X0, U_seq, X_tgt, dt_seq, t0_tensor')
if ($start -lt 0 -or $end -lt 0 -or $end -lt $start) { throw 'Helper block not found.' }
$src.RemoveRange($start, $end - $start + 1)
$newLines = @(
'def build_multistep_sequences_from_ordered_data(',
'    data_ordered,',
'    t_idx_shuffled,',
'    n_steps=5,',
'    batch_size=256,',
'    device=None,',
'    dtype=torch.float32,',
'):',
'    """',
'    Build one batch of multi-step sequences by slicing contiguous segments',
'    from the original time-ordered dataset.',
'',
'    The batch is clipped to the available valid start indices so the caller does',
'    not need to guarantee that the valid pool is larger than batch_size.',
'    """',
'    T = data_ordered.shape[0]',
'    max_t0 = (T - 1) - n_steps',
'',
'    t_valid = np.asarray(t_idx_shuffled, dtype=np.int64)',
'    t_valid = t_valid[t_valid <= max_t0]',
'    if t_valid.size == 0:',
'        raise ValueError(f"No valid start indices for n_steps={n_steps}.")',
'',
'    t0 = t_valid[: min(batch_size, len(t_valid))].astype(np.int64)',
'',
'    X0_list = []',
'    X_tgt_list = []',
'    U_seq_list = []',
'    dt_seq_list = []',
'',
'    for t in t0:',
'        seg = data_ordered[t : t + n_steps + 1]',
'        states_seg, controls_seg, dt_seg = configure_data(seg)',
'',
'        X0_list.append(states_seg[0])',
'        X_tgt_list.append(states_seg[1:])',
'        U_seq_list.append(controls_seg[:-1])',
'        dt_seq_list.append(dt_seg[:-1])',
'',
'    X0 = torch.tensor(np.stack(X0_list), dtype=dtype, device=device)',
'    X_tgt = torch.tensor(np.stack(X_tgt_list), dtype=dtype, device=device)',
'    U_seq = torch.tensor(np.stack(U_seq_list), dtype=dtype, device=device)',
'    dt_seq = torch.tensor(np.stack(dt_seq_list), dtype=dtype, device=device)',
'',
'    t0_tensor = torch.tensor(t0, dtype=torch.long, device=device)',
'    return X0, U_seq, X_tgt, dt_seq, t0_tensor'
)
for ($i = 0; $i -lt $newLines.Count; $i++) { [void]$src.Insert($start + $i, $newLines[$i]) }
$cell.source = $src.ToArray()
$json | ConvertTo-Json -Depth 200 | Set-Content -Path $path -Encoding utf8
Write-Host 'patched'