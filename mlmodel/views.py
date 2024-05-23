from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import JsonResponse
from django.contrib import messages
import pandas as pd
from .load_models import get_nb_model

def predict(request):
    if request.method == 'POST':
        required_fields = [
            'active_days', 'last_funding_year', 'first_funding_year',
            'funding_total_usd', 'founded_year', 'investment_rounds',
            'first_milestone_year', 'funding_rounds', 'country_USA',
            'milestones', 'lat', 'lng'
        ]
        
        data = {}
        for field in required_fields:
            # Using get with a default of None to handle missing fields
            value = request.POST.get(field, None)
            if value is None:
                messages.error(request, f"Missing required field: {field}")
                return redirect(reverse('mlmodel:predict'))

            try:
                value = float(value) if field != 'country_USA' else int(value)
            except ValueError:
                messages.error(request, f"Invalid input for field: {field}")
                return redirect(reverse('mlmodel:predict'))

            data[field] = value

        df = pd.DataFrame([data])
        
        try:
            model = get_nb_model()
            prediction = model.predict(df)
            prediction_status = prediction[0]
        except Exception as e:
            messages.error(request, f"An error occurred during prediction: {str(e)}")
            return redirect(reverse('mlmodel:predict'))

        # Determine the status class for styling based on the prediction
        status_class = {
            'operating': 'operating',
            'acquired': 'acquired',
            'closed': 'closed',
            'ipo': 'ipo'
        }.get(prediction_status, '')

        # Render the results page with the prediction status
        context = {
            'prediction': prediction_status,  # Here, use the key 'prediction'
            'status_class': status_class,
}
        return render(request, 'mlmodel/results.html', context)

    else:
        return show_predict_form(request)

def show_predict_form(request):
    return render(request, 'mlmodel/predict_form.html')
