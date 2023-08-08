docker build -t wayback-html-and-pdf .
docker tag wayback-html-and-pdf gcr.io/usc-research/wayback-html-and-pdf
docker push gcr.io/usc-research/wayback-html-and-pdf
gcloud builds submit --tag gcr.io/usc-research/wayback-html-and-pdf
gcloud run deploy wayback-html-and-pdf-1 \
--image=gcr.io/usc-research/wayback-html-and-pdf:latest \
--allow-unauthenticated \
--service-account=520950082549-compute@developer.gserviceaccount.com \
--memory=2Gi \
--region=us-west1 \
--project=usc-research


#for region in europe-west1 europe-west2 us-west1 us-west2 us-west3 us-east1 us-east4 us-east5 northamerica-northeast1 northamerica-northeast2
for region in us-east1 us-east4 us-east5
do
  gcloud run deploy wayback-html-and-pdf-1 \
  --image=gcr.io/usc-research/wayback-html-and-pdf:latest \
  --cpu=2 \
  --memory=8Gi \
  --region=$region \
  --project=usc-research \
   && gcloud run services update-traffic wayback-html-and-pdf-1 --to-latest --region=$region
done