# RamzlayerG — Sistem Produksi Video

**HOOK FIRST. STORY SECOND. EFFECTS WITH PURPOSE. QUALITY THROUGHOUT.**

Dokumen ini ialah standard kerja setiap video RamzlayerG (XAUUSD, market structure, price action, trading education). Toolkit `ramz/` melaksanakan sebahagian besar standard ini secara automatik; apa yang tidak boleh diautomasikan ditanda dengan jelas.

---

## 0. Aliran kerja

| Langkah | Arahan | Hasil |
|---|---|---|
| 1. Analisis rakaman | `python3 -m ramz.analyze footage/raw.mp4 --transcribe` | laporan jeda, loudness, perubahan babak, contact sheet, calon hook |
| 2. Privasi wajah | `python3 -m ramz.privacy footage/raw.mp4 -o footage/protected.mp4` | footage terlindung + `*.privacy.json` |
| 3. Tulis plan | salin `plans/example_plan.json` | susunan babak, hook, teks, anotasi chart, SFX |
| 4. Render | `python3 -m ramz.render plans/video.json` | `final.mp4` + `final.timeline.md` + `final.edl.json` |
| 5. QC | `python3 -m ramz.qc out/final.mp4 --plan plans/video.json` | semakan terukur + senarai semak manual |

Privasi dibuat **sebelum** editing supaya zoom, crop atau transisi tidak pernah mendedahkan frame yang belum dilindungi.

---

## 1. Hook engineering

- Tiada intro logo panjang. Tiada "assalamualaikum, hari ni kita nak..." sebagai pembukaan.
- Guna laporan `analyze` (transkrip + contact sheet) untuk cari ayat/visual terkuat — **utamakan ayat yang memang wujud dalam rakaman**.
- Jika hook terbaik di tengah/hujung rakaman, letak ia sebagai segment pertama (`"role": "hook"`). Segment seterusnya kembali ke susunan asal.
- Jangan ubah konteks sehingga maksud jadi mengelirukan. Jangan reka dakwaan untung, keputusan trade atau fakta.
- Sasaran: hook ≤ 3s. `render` dan `qc` memberi amaran jika segment pertama bukan hook atau melebihi had.
- Pelbagaikan gaya hook antara video:

| Jenis hook | Bila guna | Rawatan visual/audio |
|---|---|---|
| Kesilapan biasa | "Ramai trader salah baca BOS" | tight zoom pada candle + soft impact |
| Soalan | "Kenapa SL kau selalu kena sebelum harga naik?" | teks headline slide-up, tiada SFX berat |
| Visual reveal | chart terus tunjuk liquidity sweep | push-in perlahan + whoosh lembut |
| Kenyataan tegas (benar) | "Breakout bukan signal entry." | flash halus + bass hit |
| Hasil pengajaran | tunjuk zon yang dipertahankan dahulu, terangkan kemudian | zone reveal + ui blip |

## 2. Sound design

Kit asas dijana dengan `python3 -m ramz.sfx` → `assets/sfx/` (whoosh, reverse_whoosh, soft_impact, bass_hit, riser, click, ui_blip, typing, glitch, ambient_bed). Kit ini sintetik dan bersih; untuk video utama, ganti dengan pustaka berlesen yang lebih kaya (nama fail sama).

| Momen | SFX dicadang | Gain |
|---|---|---|
| Ayat hook muncul | soft_impact / bass_hit | −12 dB |
| Chart/zon didedahkan | whoosh | −16 hingga −20 dB |
| Label HH/HL/LH/LL | ui_blip | −18 dB |
| Transisi babak | whoosh / reverse_whoosh | −16 dB |
| Penutup | riser lembut (pilihan) | −22 dB |

Automatik dalam `render`: high-pass 80 Hz, EQ suara ringan, micro-fade pada setiap potongan (tiada klik), muzik di-*duck* bila suara masuk (sidechain), loudness 2-pass ke −14 LUFS, true peak ≤ −1 dBTP.

Peraturan: maksimum ~1 SFX setiap 3 saat secara purata (`qc` menyemak kepadatan). Jangan guna glitch pada setiap transisi.

## 3. Motion

- `zoom` menggunakan penapis `perspective` dengan sampling sub-pixel — push-in perlahan tanpa getaran.
- Setiap zoom mesti ada fungsi: tunjuk bahagian chart yang sedang diterangkan (`cx`, `cy` = titik fokus).
- Elak zoom pada setiap segment. Push-in (1.0→1.1–1.2) untuk penekanan, pull-back (1.08→1.0) untuk penutup/rumusan.
- Speed ramp, whip pan, parallax dan motion blur berarah **tidak** dibuat oleh toolkit ini — perlu editor (DaVinci Resolve / Premiere / After Effects / CapCut).

## 4. Special effects (terkawal)

| fx | Fungsi | Had |
|---|---|---|
| `flash` | pop exposure halus pada impact hook | ≤ 0.15s, amount ≤ 0.2 |
| `glow` | bloom lembut pada highlight | opacity ≤ 0.3, jangan pada chart padat |
| `rgb_split` | hit kromatik sangat singkat | ≤ 0.12s, maksimum sekali setiap video |
| `blur_in` | focus pull dari lembut ke tajam | permulaan babak sahaja |
| vignette | fokus ke tengah | 0.2–0.4 |

Light leak, particles dan tracking-based highlights memerlukan editor/plugin luar.

## 5. Color grading

Preset dalam `render` (sengaja lembut supaya warna candle bull/bear tidak berubah):

| Preset | Sesuai untuk |
|---|---|
| `dark_cinematic` | identiti utama RamzlayerG, chart tema gelap |
| `neutral` | screen recording yang sudah bersih |
| `cool_modern` | B-roll meja/setup, tema biru |
| `clean_contrast` | rakaman pudar / terlebih dedah |
| `natural` | rakaman kamera dengan kulit/tangan |
| `none` | bila footage sudah betul |

Satu preset untuk keseluruhan video (`qc` memberi amaran jika bercampur). Overlay teks & anotasi dilukis **selepas** grade supaya warna jenama tepat.

## 6. Animasi & motion graphics

Teks (`text`): gaya `headline`, `emphasis`, `caption`, `label`, `lower_third`; animasi `slide_up`, `pop`, `fade`, `slide_left`; fade masuk/keluar automatik.

Anotasi chart (`annotations`) — koordinat dinormalkan pada kanvas output:

| type | Kegunaan |
|---|---|
| `level` | garisan S/R dilukis kiri→kanan + label |
| `zone` | supply/demand didedahkan berperingkat + label |
| `label` | HH / HL / LH / LL / BOS / CHoCH |
| `arrow` | arah pergerakan yang diterangkan |

**Kedudukan setiap anotasi mesti diambil dari chart sebenar** (jeda video pada frame tersebut dan ukur). Jangan reka pergerakan harga. Ilustrasi mesti jelas bukan rakaman pasaran sebenar.

## 7. Privasi wajah

`ramz.privacy` (dua laluan):
1. Pengesanan Haar (depan + profil kiri/kanan) pada setiap frame ke-2.
2. Jurang antara pengesanan diisi dengan interpolasi **dua arah** (tiada lag bila kepala bergerak, tiada lubang bila wajah berpaling), kemudian dilicinkan secara berpusat; saiz mask membesar serta-merta, mengecil perlahan, dan melebar bila kepala bergerak.
3. Blur berbulu (feathered). Footage HDR iPhone (HLG/PQ) ditukar ke SDR BT.709 dan VFR ditukar ke 30fps CFR.
4. Audit: untuk setiap frame yang ada wajah dikesan — kotak wajah mesti berada di dalam mask, dan butiran (edge) yang tinggal selepas blur mesti < 15% daripada asal.

- `--method soft` (default) · `pixel` (lebih kuat) · `solid` (paling selamat).
- `--regions manual.json` — kawasan keyframe manual sentiasa digunakan di atas pengesanan automatik. Untuk set-up meja yang tetap, `--no-detect --regions` paling pantas dan paling boleh dipercayai.
- Laporan `*.privacy.json` menyenaraikan julat masa yang perlu disemak manusia. `render` menolak plan `privacy_required: true` tanpa laporan ini.
- Pengesanan automatik **bukan jaminan 100%**. Tonton setiap julat `review_needed` sebelum publish. Privasi didahulukan berbanding estetika.

## 8. Pembersihan percakapan

- `analyze --transcribe` (Whisper `medium`, Bahasa Melayu) memberi transkrip peringkat perkataan dengan keyakinan setiap perkataan, jeda, dan filler words.
- Potongan ditentukan dengan sempadan pertuturan sebenar (timestamp perkataan + envelope tenaga audio), bukan anggaran.
- Timestamp perkataan Whisper sering 'hanyut' ke dalam jeda. Bila teks setiap segment diketahui, isi medan `text` pada segment dan jalankan `python3 -m ramz.align plan.json -o x.forced.json` (MMS forced alignment) — guna fail itu sebagai `captions.transcript`, dan set `drop_below: 0` kerana teks sudah disahkan.
- Caption automatik (`captions` dalam plan) dipetakan semula ke timeline akhir selepas potongan. Perkataan yang tidak jelas **tidak dipaparkan** sehingga disahkan (`omit`), pembetulan disenaraikan (`replace`), dan semuanya ditulis ke `*.captions_review.md`.
- Potong dengan memecahkan rakaman kepada segment; simpan ~0.25s jeda natural.
- Kekalkan intonasi, loghat Melayu Malaysia, penekanan dan rentak. Jangan buat terlalu laju/sempurna.
- Tiada AI voice tanpa arahan khusus. Ayat yang tidak jelas → tanda untuk pengesahan, jangan reka.

## 9. Pacing

Hook → konteks → isi → takeaway → penutup. Tiada pengulangan, tiada efek pada setiap ayat, CTA hanya bila sesuai.

## 10. Timeline

`render` menulis `*.timeline.md` secara automatik untuk setiap babak: visual, teks, animasi chart, motion, FX, audio/SFX, warna, transisi, dan tujuan kreatif (medan `purpose`). Rujuk `docs/TIMELINE_TEMPLATE.md` untuk format perancangan sebelum render.

## 11. QC

`ramz.qc` mengukur: resolusi, CFR, frame hitam/beku, loudness, true peak, clipping, hook di depan, panjang hook, kepadatan efek, konsistensi grade, status privasi. Perkara subjektif (kekuatan hook, ketepatan trading, keaslian suara) disenaraikan sebagai semakan manual — **tiada skor 10/10 automatik**.

## 12. Apa yang toolkit ini tidak buat

Supaya jujur tentang had:

- Tiada akses langsung ke Google Drive — muat turun footage ke `footage/` dahulu.
- Speed ramp, whip pan, parallax, light leak, particles, tracking highlights → editor/plugin luar.
- Transkrip automatik memerlukan `pip install faster-whisper` + muat turun model (~1.5GB untuk `medium`).
- Pemotongan filler word dibuat melalui segment dalam plan (separuh manual), bukan auto-delete.
- Pengesanan wajah Haar boleh terlepas wajah sisi/gelap/kabur — sebab itu ada `--regions` dan laporan semakan.
