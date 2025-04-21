document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analysisForm');
    const fileInput = document.getElementById('file');
    const preview = document.getElementById('previewImage');
    const resultDiv = document.getElementById('result');
    const themeToggle = document.getElementById('themeToggle');

    // 图片预览
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = e => {
                preview.src = e.target.result;
                preview.classList.remove('d-none');
            };
            reader.readAsDataURL(file);
        } else {
            preview.classList.add('d-none');
        }
    });

    // 表单提交处理
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        resultDiv.innerHTML = '<p>⏳ 分析中...</p>';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            resultDiv.classList.add('fade-in');
            resultDiv.innerHTML = '';

            if (result.image) {
                resultDiv.innerHTML += `<h4>🖼 图像分析结果</h4><ul>`;
                for (let [key, value] of Object.entries(result.image.theme_probs)) {
                    resultDiv.innerHTML += `<li>${key}：${value}</li>`;
                }
                resultDiv.innerHTML += `<li>主主题：<strong>${result.image.main_theme}</strong></li>`;
                resultDiv.innerHTML += `<li>情感：<strong>${result.image.sentiment}</strong></li></ul>`;
            }

            if (result.text) {
                resultDiv.innerHTML += `<h4>📝 文本分析结果</h4><ul>`;
                for (let [key, value] of Object.entries(result.text)) {
                    resultDiv.innerHTML += `<li>${key}：${(value * 100).toFixed(1)}%</li>`;
                }
                resultDiv.innerHTML += `</ul>`;
            }

        } catch (error) {
            console.error('提交失败:', error);
            resultDiv.innerHTML = '<p class="text-danger">❌ 分析失败，请重试！</p>';
        }
    });

    // 主题切换
    themeToggle.addEventListener('click', () => {
        const body = document.body;
        const isLight = body.classList.contains('light-mode');

        body.classList.toggle('light-mode', !isLight);
        body.classList.toggle('dark-mode', isLight);
        themeToggle.textContent = isLight ? '☀️ 日间模式' : '🌙 夜间模式';
    });
});
