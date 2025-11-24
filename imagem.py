import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fourier & Fase Aleatória", page_icon="📷", layout="wide")
st.title("📷 Transformada de Fourier com Fase Aleatória")

st.markdown(
    """
Este app:
1. Carrega uma imagem.
2. Calcula a Transformada de Fourier 2D.
3. Mantém o módulo e substitui a **fase** por valores aleatórios.
4. Reconstrói a imagem a partir desse novo espectro.

Isso mostra como a **fase** é fundamental para a estrutura da imagem.
"""
)

# ------------------------------
# Funções auxiliares
# ------------------------------
def carregar_imagem(arquivo, modo="L"):
    """Carrega a imagem e converte para o modo especificado (L = escala de cinza)."""
    img = Image.open(arquivo)
    img = img.convert(modo)  # 'L' = grayscale
    return img

def fourier_random_phase(img_array, seed=None):
    """
    Recebe uma imagem em escala de cinza (array 2D),
    calcula a FFT, substitui a fase por valores aleatórios e reconstrói.
    """
    if seed is not None:
        np.random.seed(seed)

    # FFT 2D
    F = np.fft.fft2(img_array)
    mag = np.abs(F)           # módulo
    # phase = np.angle(F)     # fase original (não vamos usar agora, mas poderia guardar)

    # Fase aleatória uniforme em [-pi, pi]
    random_phase = np.random.uniform(-np.pi, np.pi, size=F.shape)

    # Novo espectro: mesmo módulo, fase aleatória
    F_rand = mag * np.exp(1j * random_phase)

    # Reconstruir imagem
    img_rec = np.fft.ifft2(F_rand)
    img_rec = np.real(img_rec)

    # Normalizar para 0-255
    img_rec = img_rec - img_rec.min()
    if img_rec.max() > 0:
        img_rec = img_rec / img_rec.max()
    img_rec = (img_rec * 255).astype(np.uint8)

    return img_rec

def calcular_espectro_magnitude(img_array):
    """Retorna o espectro de magnitude (log) para visualização."""
    F = np.fft.fft2(img_array)
    F_shift = np.fft.fftshift(F)
    mag = np.abs(F_shift)
    # Escala log para visualização
    mag_log = np.log1p(mag)
    mag_log = mag_log / mag_log.max()
    mag_log = (mag_log * 255).astype(np.uint8)
    return mag_log

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("Configurações")

arquivo = st.sidebar.file_uploader(
    "Carregue uma imagem (JPG, PNG, etc.)", 
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

use_seed = st.sidebar.checkbox("Usar semente fixa para fase aleatória?", value=False)
seed_value = None
if use_seed:
    seed_value = st.sidebar.number_input("Semente (seed)", min_value=0, max_value=10_000, value=42, step=1)

mostrar_espectro = st.sidebar.checkbox("Mostrar espectro de magnitude (FFT)", value=True)

# ------------------------------
# Corpo principal
# ------------------------------
if arquivo is None:
    st.info("👈 Carregue uma imagem na barra lateral para começar.")
else:
    # Carregar imagem
    img = carregar_imagem(arquivo, modo="L")
    img_np = np.array(img).astype(float)

    # Calcular imagem reconstruída com fase aleatória
    img_rec_np = fourier_random_phase(img_np, seed=seed_value)

    # Calcular espectro de magnitude (opcional)
    if mostrar_espectro:
        mag_log = calcular_espectro_magnitude(img_np)

    # Layout em colunas
    cols = st.columns(3 if mostrar_espectro else 2)

    with cols[0]:
        st.subheader("Imagem original (escala de cinza)")
        st.image(img, use_container_width=True, clamp=True)

    if mostrar_espectro:
        with cols[1]:
            st.subheader("Espectro de magnitude (log)")
            st.image(mag_log, use_container_width=True, clamp=True)

        with cols[2]:
            st.subheader("Reconstrução com fase aleatória")
            st.image(img_rec_np, use_container_width=True, clamp=True)
    else:
        with cols[1]:
            st.subheader("Reconstrução com fase aleatória")
            st.image(img_rec_np, use_container_width=True, clamp=True)

    st.markdown("---")
    st.markdown(
        """
**Resumo matemático:**

Seja a imagem \( f(x,y) \) e sua FFT dada por

\\[
F(u,v) = A(u,v) e^{j \\phi(u,v)}
\\]

onde \( A(u,v) = |F(u,v)| \) é o módulo e \( \\phi(u,v) \) é a fase.

Neste app, mantemos \( A(u,v) \) e substituímos \( \\phi(u,v) \) por uma fase aleatória
uniforme em \([-\\pi, \\pi]\). A imagem reconstruída é:

\\[
\\tilde{f}(x,y) = \\mathcal{F}^{-1}\\{ A(u,v) e^{j \\phi_{rand}(u,v)} \\}
\\]

Isso destrói a estrutura espacial da imagem, mostrando o papel crucial da fase.
"""
    )
