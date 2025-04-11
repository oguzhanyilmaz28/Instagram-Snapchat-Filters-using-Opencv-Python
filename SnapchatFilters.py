"""
   This program is an image processing application that 
   takes camera images and applies different TikTok filters. 
   Users can switch between different modes.
"""

import cv2
import numpy as np
import math
import time

def create_tiktok_filter(frame, frame_count=0, mode=0):
    """
    Applies a TikTok-like filter to the given frame based on the selected mode.
    Parameters:
        frame: Video frame to be processed
        frame_count: Frame counter for animation
        mode: Filter mode to be applied (0-5)
    Return:
        Video frame with filter applied
    """
    # Performans için görüntüyü küçült
    height, width = frame.shape[:2]
    scale_factor = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    small_height, small_width = small_frame.shape[:2]
    
    result = small_frame.copy()
    
    # Merkez koordinatlar
    center_x, center_y = small_width // 2, small_height // 2
    
    # 1. Mod bazlı zaman faktörü - animasyon hızını belirler
    time_factors = {
        0: frame_count / 25.0,     # Temel mod - daha yavaş
        1: frame_count / 15.0,     # Dalga+ mod - orta hız
        2: frame_count / 10.0,     # Su mod 
        3: frame_count / 20.0,     # Neon mod 
        4: frame_count / 30.0,     # Retro mod 
        5: int(frame_count / 3) * 0.5  # Glitch mod - kesintili 
    }
    time_factor = time_factors[mode]
    
    # 2. Mod bazlı renk kayma parametreleri - RGB kanallarının ne kadar kayacağını belirler
    shift_amounts = {
        0: 3,    # Temel mod
        1: 5,    # Dalga+ mod
        2: 7,    # Su mod 
        3: 5,    # Neon mod 
        4: 2,    # Retro mod 
        5: 12 if frame_count % 15 < 3 else 4  # Glitch mod - kesintili
    }
    shift_amount = shift_amounts[mode]
    
    # 3. Mod bazlı dalga parametreleri - dalga efektlerinin genlik, frekans ve fazları
    wave_params_dict = {
        0: [  # Temel mod - daha basit dalgalar
            (3, 20, time_factor * 0.5),  # (genlik, frekans, faz)
        ],
        1: [  # Dalga+ mod - çoklu dalgalar
            (4, 10, time_factor * 0.5),
            (6, 15, time_factor * 0.7),
            (8, 20, time_factor * 0.9),
        ],
        2: [  # Su mod - su yüzeyindeki dalgalanma efekti
            (4 + math.sin(frame_count / 20.0) * 2, 15, time_factor * 0.7),
            (2.4 + math.sin(frame_count / 20.0) * 1.2, 25, time_factor * 0.5 + math.pi),
            (1.2 + math.sin(frame_count / 20.0) * 0.6, 40, time_factor * 0.3),
        ],
        3: [  # Neon mod - ışık dalgası efekti
            (3, 30, time_factor * 0.3),  # Daha uzun dalgalar
            (2, 50, time_factor * 0.2),
        ],
        4: [  # Retro mod - minimum dalgalanma
            (2, 40, time_factor * 0.3),  # Çok hafif dalgalar
        ],
        5: [  # Glitch mod - rastgele, kesintili dalgalar
            (0, 1, 0) if frame_count % 10 < 6 else (10, 8, time_factor * 2.0),
            (8, 12, time_factor * 1.5) if frame_count % 10 >= 6 else (0, 1, 0)
        ]
    }
    wave_params = wave_params_dict[mode]
    
    # 4. Mod bazlı parlaklık ve kontrast ayarları
    brightness_contrast = {
        0: (15, 1.0),    # Temel mod - normal
        1: (20, 1.1),    # Dalga+ mod - biraz daha parlak
        2: (18, 1.15),   # Su mod 
        3: (30, 1.25),   # Neon mod
        4: (10, 0.95),   # Retro mod 
        5: (15, 1.2)     # Glitch mod 
    }
    
    # 5. Mod bazlı vignette yoğunluğu - kenarların kararma efekti
    vignette_intensities = {
        0: -0.2,   # Temel mod
        1: -0.3,   # Dalga+ mod
        2: -0.4,   # Su mod 
        3: -0.6,   # Neon mod 
        4: -0.5,   # Retro mod 
        5: -0.3 - (math.sin(frame_count / 10.0) * 0.2)  # Glitch mod 
    }
    
    # RENK KAYMA EFEKTİ - RGB kanallarını ayrıştırıp kaydırarak kromatik aberasyon
    b, g, r = cv2.split(small_frame)
    
    # Mod bazlı renk düzenlemeleri
    if mode == 0:  # Temel mod
        pass  # Temel renk değişikliği yok
    elif mode == 1:  # Dalga+ mod
        pass  # Temel renk değişikliği yok
    elif mode == 2:  # Su mod
        # Pulse hesaplama - dalgalanan bir kayma miktarı
        pulse = math.sin(frame_count / 15.0) * 3 + 4  # 1-7 arası değişen değer
        shift_amount = int(pulse)
    elif mode == 3:  # Neon mod
        # Kanalları ayrı ayrı parlatma
        glow_amount = int(math.sin(frame_count / 10.0) * 2 + 3)
        r = cv2.addWeighted(r, 1, r, 0.5, glow_amount)
        g = cv2.addWeighted(g, 1, g, 0.5, glow_amount - 2)
        b = cv2.addWeighted(b, 1, b, 0.5, glow_amount - 1)
    elif mode == 4:  # Retro mod
        # Nostaljik film efekti - kırmızı vurgulu, mavi ve yeşil azaltılmış
        r = cv2.convertScaleAbs(r, alpha=1.2, beta=5)
        b = cv2.convertScaleAbs(b, alpha=0.9, beta=0)
        g = cv2.convertScaleAbs(g, alpha=0.85, beta=0)
    
    # Kaydırma matrisleri oluştur
    M_r = np.float32([[1, 0, shift_amount], [0, 1, 0]])
    M_b = np.float32([[1, 0, -shift_amount], [0, 1, 0]])
    
    # Kırmızı ve mavi kanalları zıt yönlerde kaydır
    r_shifted = cv2.warpAffine(r, M_r, (small_width, small_height))
    b_shifted = cv2.warpAffine(b, M_b, (small_width, small_height))
    
    # Kanalları birleştir
    result = cv2.merge([b_shifted, g, r_shifted])
    
    # MOD BAZLI EKSTRA RENK EFEKTLERİ
    if mode == 2:  # Su modu
        # Renk vurgulama - sudan yansıma efekti için
        result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(result_hsv)
        # Doygunluğu artır, ama daha doğal
        s = cv2.convertScaleAbs(s, alpha=1.2, beta=0)
        v = cv2.convertScaleAbs(v, alpha=1.1, beta=5)
        result_hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
    
    elif mode == 3:  # Neon modu
        # Kenarları tespit et
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None)
        
        # Kenarları renklendir - neon ışık efekti için
        edge_color = np.zeros_like(result)
        
        # Zaman bazlı renk değişimi - yanıp sönen neon efekti
        hue = (frame_count * 2) % 180
        edge_hsv = np.zeros((small_height, small_width, 3), dtype=np.uint8)
        edge_hsv[:, :, 0] = hue  # H - ton
        edge_hsv[:, :, 1] = 255  # S - doygunluk
        edge_hsv[:, :, 2] = 255  # V - parlaklık
        edge_color = cv2.cvtColor(edge_hsv, cv2.COLOR_HSV2BGR)
        
        # Kenarları aydınlat - neon parlaması efekti
        edge_mask = cv2.merge([edges, edges, edges]) / 255.0
        result = cv2.addWeighted(result, 1, edge_color, 0.6, 0) * (1 - edge_mask * 0.5) + edge_color * edge_mask * 0.5
        
        # Genel parlaklık artırma
        result = cv2.convertScaleAbs(result, alpha=1.2, beta=15)
    
    elif mode == 4:  # Retro mod
        # VHS tarzı noise efekti - eski video görüntüsü için
        noise = np.zeros((small_height, small_width), dtype=np.uint8)
        cv2.randn(noise, 0, 25)  # Gaussian noise
        noise_color = cv2.merge([noise, noise, noise])
        result = cv2.add(result, noise_color)
        
        # Dikey çizgiler (VHS tracking lines) - eski video kasetlerdeki gibi
        if frame_count % 30 < 5:  # Her 30 karede bir tracking sorunu
            line_pos = np.random.randint(0, small_height)
            line_height = np.random.randint(5, 20)
            result[line_pos:line_pos+line_height, :] = result[line_pos:line_pos+line_height, :] * 0.5
        
        # Sıcak vintage renk tonu
        result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(result_hsv)
        # Sarımsı tonlara kaydır
        h = cv2.add(h, 10) % 180
        # Hafif solgunluk
        s = cv2.convertScaleAbs(s, alpha=0.8, beta=0)
        result_hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
    
    elif mode == 5:  # Glitch mod
        # Rastgele blok kaymaları - dijital bozulma efekti
        if frame_count % 20 < 3:  # Arada sırada glitch efekti
            # Rastgele blok seçimi
            block_height = np.random.randint(10, small_height // 8)
            block_y = np.random.randint(0, small_height - block_height)
            
            max_shift = small_width // 2
            block_shift = np.random.randint(-max_shift, max_shift)
            
            # Bloğu kaydır (güvenli şekilde)
            if block_shift > 0:
                if block_shift < small_width:
                    result[block_y:block_y+block_height, block_shift:] = result[block_y:block_y+block_height, :-block_shift]
                    result[block_y:block_y+block_height, :block_shift] = 0
            else:
                block_shift = abs(block_shift)
                if block_shift < small_width:
                    result[block_y:block_y+block_height, :-block_shift] = result[block_y:block_y+block_height, block_shift:]
                    result[block_y:block_y+block_height, -block_shift:] = 0
        
        # Piksel bozulması - bloklu görüntü efekti
        if frame_count % 15 < 2:
            pixel_block = np.random.randint(2, 8)  # Piksel blok boyutu
            for y in range(0, small_height, pixel_block):
                for x in range(0, small_width, pixel_block):
                    y_end = min(y + pixel_block, small_height)
                    x_end = min(x + pixel_block, small_width)
                    block = result[y:y_end, x:x_end]
                    if len(block) > 0 and len(block[0]) > 0:  # Boş blok kontrolü
                        color = block[0, 0].copy()
                        result[y:y_end, x:x_end] = color
        
    # DALGA EFEKTİ - görüntüyü dalgalandırarak efekt oluşturma
    # Daha hızlı hesaplama için vektörize edilmiş yaklaşım
    y_indices, x_indices = np.mgrid[0:small_height, 0:small_width]
    
    # Merkeze olan mesafeyi hesaplama
    dx = x_indices - center_x
    dy = y_indices - center_y
    distance = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # Çoklu dalga efektlerini birleştirme
    x_offset = np.zeros_like(distance, dtype=np.float32)
    y_offset = np.zeros_like(distance, dtype=np.float32)
    
    # Temel dalga efektleri
    for amplitude, frequency, phase in wave_params:
        # Radyal dalga - merkezden dışa doğru
        radial_offset = amplitude * np.sin(distance / frequency + phase)
        
        # X ve Y bileşenlerine ayrıştırma
        x_offset += radial_offset * np.cos(angle)
        y_offset += radial_offset * np.sin(angle)
        
        # Spiral efekti (Dalga+ mod için)
        if mode == 1:
            spiral_factor = 0.02
            spiral_x = amplitude * 0.5 * np.sin(angle * 2 + phase) * np.exp(-distance * spiral_factor)
            spiral_y = amplitude * 0.5 * np.cos(angle * 2 + phase) * np.exp(-distance * spiral_factor)
            
            x_offset += spiral_x
            y_offset += spiral_y
    
    # Mod özel dalga efektleri
    if mode == 2:  # Su modu için özel dalga efekti
        # Yüzey dalgalanması efekti - yatay dalgalar
        wave_height = 3.5
        wave_x = np.sin(y_indices / 10 + time_factor) * wave_height
        wave_y = np.cos(x_indices / 10 + time_factor * 0.7) * wave_height * 0.5
        
        x_offset += wave_x
        y_offset += wave_y
        
        # Ekstra su efekti - kenardan merkeze doğru dalgalanma
        edge_wave = 2 * np.sin(distance / 30 - time_factor * 0.5) * np.exp(-distance * 0.01)
        x_offset += edge_wave * np.cos(angle)
        y_offset += edge_wave * np.sin(angle)
    
    # Harita koordinatlarını güncelleme - piksellerin yeni konumları
    map_x = x_indices + x_offset
    map_y = y_indices + y_offset
    
    # Sınırları kontrol etme - taşmaları engelle
    map_x = np.clip(map_x, 0, small_width - 1)
    map_y = np.clip(map_y, 0, small_height - 1)
    
    # Haritayı kullanarak yeniden örnekleme - pikselleri yeni konumlara taşı
    result = cv2.remap(result, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)
    
    # VIGNETTE EFEKTİ - kenarları karartan oval maske
    mask = np.zeros((small_height, small_width), dtype=np.uint8)
    cv2.ellipse(mask, (center_x, center_y), (small_width//2, small_height//2), 0, 0, 360, 255, -1)
    
    # Çekirdek boyutlarını tek sayı olacak şekilde ayarla
    kernel_width = max(3, small_width//5 + (1 if small_width//5 % 2 == 0 else 0))
    kernel_height = max(3, small_height//5 + (1 if small_height//5 % 2 == 0 else 0))
    
    # Maskeyi yumuşat - daha doğal bir geçiş için
    mask = cv2.GaussianBlur(mask, (kernel_width, kernel_height), 0)
    
    # Vignette yoğunluğunu uygula
    vignette_intensity = vignette_intensities[mode]
    result = cv2.addWeighted(result, 1, cv2.merge([mask//3, mask//3, mask//3]), vignette_intensity, 0)
    
    # PARLAKLIL VE KONTRAST AYARI
    brightness, contrast = brightness_contrast[mode]
    
    # Mod özel düzenlemeler
    if mode == 2:  # Su mod için ekstra parlama efekti
        if frame_count % 30 < 5:  # Her 30 karede bir parlama efekti
            brightness += 10
            contrast += 0.1
    elif mode == 3:  # Neon mod için yanıp sönme efekti
        flash_intensity = math.sin(frame_count / 5.0) * 10
        brightness += int(flash_intensity)
    elif mode == 5:  # Glitch mod - rastgele parlaklık değişimleri
        if frame_count % 8 < 2:
            brightness += np.random.randint(-20, 30)
            contrast += np.random.random() * 0.3 - 0.1
    
    # Parlaklık ve kontrast ayarlarını uygula
    result = cv2.convertScaleAbs(result, alpha=contrast, beta=brightness)
    
    # MOD BAZLI SON DÜZENLEMELER
    if mode == 4:  # Retro mod 
        # Hafif yatay çizgiler - CRT ekran efekti
        for y in range(0, small_height, 3):
            result[y, :] = result[y, :] * 0.85
    
    elif mode == 5:  # Glitch mod 
        # Renk kanalı kaymaları - dijital sinyal bozulması efekti
        if frame_count % 12 < 3:
            rgb_channels = cv2.split(result)
            channel_idx = np.random.randint(0, 3)
            
            shift_y = np.random.randint(-4, 5)  # -4 ile 4 arası
            
            if shift_y != 0:
                if shift_y > 0:
                    rgb_channels[channel_idx][shift_y:, :] = rgb_channels[channel_idx][:-shift_y, :]
                    rgb_channels[channel_idx][:shift_y, :] = 0
                else:
                    shift_y = abs(shift_y)
                    rgb_channels[channel_idx][:-shift_y, :] = rgb_channels[channel_idx][shift_y:, :]
                    rgb_channels[channel_idx][-shift_y:, :] = 0
                
            result = cv2.merge(rgb_channels)
    
    # Sonucu orijinal boyuta geri getirme
    result = cv2.resize(result, (width, height))
    
    return result

def main():
    """
    It's main function is to launch the camera and apply filters.
    """
    # Kamera yakalama nesnesini oluştur
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera could not be opened!")
        return
    
    # Kamera çözünürlüğünü ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS sayacı ve diğer değişkenleri başlat
    fps_counter = 0
    fps = 0
    fps_start_time = time.time()
    frame_count = 0
    
    # Filtre modları listesi
    filter_modes = ["Basic", "Wawe+", "Water", "Neon", "Retro", "Glitch"]
    current_mode = 0
    
    # Ana döngü
    while True:
        # Kameradan görüntü al
        ret, frame = cap.read()
        if not ret:
            print("Image not received!")
            break
            
        # Seçili filtreyi uygula
        filtered_frame = create_tiktok_filter(frame, frame_count, current_mode)
        
        # Frame sayacını arttır
        frame_count = (frame_count + 1) % 10000
        
        # FPS hesaplama
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Görüntü boyutlarını al
        height, width = filtered_frame.shape[:2]
        
        # FPS ve mod bilgisini ekrana yazdır
        cv2.putText(filtered_frame, f"FPS: {fps}", (10, 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(filtered_frame, f"Mod: {filter_modes[current_mode]}", (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Alt bilgi (Sağ alt köşe - Dinamik konumlandırma)
        text = "M: Change Mode, Q: Quit"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = width - text_size[0] - 10  # Sağdan 10px boşluk
        text_y = height - 10  # Alttan 10px boşluk
        cv2.putText(filtered_frame, text, (text_x, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Görüntüyü göster
        cv2.imshow('Snapchat Filter', filtered_frame)
        
        # Tuş kontrolü
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            current_mode = (current_mode + 1) % len(filter_modes)
    
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

# Program başlangıcı
if __name__ == "__main__":
    main()