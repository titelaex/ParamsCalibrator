# Deploy pe AWS EC2 — ghid de referință

Acoperă partea de AWS/EC2 pentru microserviciul de calibrare (FastAPI + Docker).
Pașii 1-4 (cont AWS, instanță, Docker, cod pe instanță) pot fi făcuți oricând,
independent de stadiul codului. Pașii 5-6 presupun că `src/api/` conține deja
un `Dockerfile` funcțional care încapsulează modelul MLP antrenat — de
completat cu comenzile exacte odată ce acel milestone e gata.

## 1. Cont AWS + instanță EC2

Din consola AWS: EC2 → Launch Instance.

- **AMI:** Amazon Linux 2023 sau Ubuntu 22.04 (ambele au imagini gratuite Free Tier)
- **Tip instanță:** `t2.micro` / `t3.micro` — suficient pentru un MLP mic servit prin FastAPI
- **Key pair:** creezi una nouă, descarci `.pem`-ul (necesar pentru SSH)
- **Security group:** inbound 22 (SSH, ideal restricționat la IP-ul tău) + inbound 80 (HTTP, `0.0.0.0/0` pentru demo)

Free Tier: 750 ore/lună gratuit pe `t2.micro`/`t3.micro`, primele 12 luni pentru conturi noi.

## 2. Conectare SSH

```
chmod 400 numele-cheii.pem
ssh -i numele-cheii.pem ec2-user@<IP-PUBLIC-INSTANTA>
```

(`ec2-user` pentru Amazon Linux; `ubuntu` pentru Ubuntu)

## 3. Instalare Docker (Amazon Linux 2023)

```
sudo dnf install docker -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user
# deconectează-te și reconectează-te prin SSH ca să aibă efect usermod
```

## 4. Codul pe instanță

Opțiunea simplă — git clone (dacă proiectul e pe GitHub):

```
git clone <url-repo>
```

Alternativ, copiere directă de pe laptop:

```
scp -i numele-cheii.pem -r ParamsCalibrator ec2-user@<IP-PUBLIC>:~/
```

## 5. Build & run container [DE COMPLETAT după ce există Dockerfile-ul]

```
docker build -t paramscalibrator-api .
docker run -d -p 80:8000 paramscalibrator-api
```

## 6. Testare

```
curl localhost/docs        # direct pe instanță
```

Din orice browser: `http://<IP-PUBLIC-INSTANTA>/docs` (Swagger UI generat automat de FastAPI) — acesta e link-ul de arătat live în prezentare.

## 7. Pentru ziua prezentării

- Alocă o **Elastic IP** instanței (gratuit cât timp e atașată unei instanțe pornite), ca adresa să nu se schimbe dacă repornești instanța.
- **Oprește sau termină instanța după prezentare** — evită costuri reziduale în afara Free Tier.
