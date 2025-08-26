document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const resultText = document.getElementById('result-text');
    const imagePreview = document.getElementById('image-preview');
    const locationBtn = document.getElementById('location-btn');
    const locationInfo = document.getElementById('location-info');

    // Handle file input change to show a preview
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    // Handle form submission for classification
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
            resultText.textContent = 'Пожалуйста, выберите файл для загрузки.';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Reset result text
        resultText.textContent = 'Классификация...';

        try {
            const response = await fetch('/classify', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Ошибка сервера: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            resultText.textContent = `Тип мусора: ${data.classification}`;

        } catch (error) {
            console.error('Ошибка при классификации:', error);
            resultText.textContent = `Ошибка: ${error.message}`;
        }
    });

    // Handle geolocation button click
    locationBtn.addEventListener('click', () => {
        if (!navigator.geolocation) {
            locationInfo.textContent = 'Геолокация не поддерживается вашим браузером.';
            return;
        }

        locationInfo.textContent = 'Определение местоположения...';

        navigator.geolocation.getCurrentPosition(
            (position) => {
                const { latitude, longitude } = position.coords;
                locationInfo.innerHTML = `
                    <strong>Ваши координаты:</strong><br>
                    Широта: ${latitude.toFixed(6)}<br>
                    Долгота: ${longitude.toFixed(6)}
                `;
            },
            (error) => {
                let message = 'Не удалось получить геолокацию.';
                switch (error.code) {
                    case error.PERMISSION_DENIED:
                        message = 'Вы запретили доступ к геолокации.';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        message = 'Информация о местоположении недоступна.';
                        break;
                    case error.TIMEOUT:
                        message = 'Время ожидания запроса геолокации истекло.';
                        break;
                }
                locationInfo.textContent = message;
                console.error('Ошибка геолокации:', error);
            }
        );
    });
});
