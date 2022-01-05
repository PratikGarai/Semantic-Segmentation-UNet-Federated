## Commands for Chennai Dataset :

```sh
python LogParserFederated.py --file DataExamples\clientCHN1.txt --name CHN1 --sep "Fiting started on Client..."
```

```sh
python LogParserFederated.py --file DataExamples\clientCHN2.txt --name CHN2 --sep "Fiting started on Client..."
```

```sh
python LogParserFederated.py --file DataExamples\clientCHN3.txt --name CHN3 --sep "Fiting started on Client..."
```

```sh
python LogParserUnified.py --file DataExamples\UnifiedCHN.txt --name CHN
```

```sh
python PlotterDriver.py --name1 CHN1 --name2 CHN2 --name3 CHN3 --nameU CHN --savedir CHN
```

## Commands for Cityscape Dataset :

```sh
python LogParserFederated.py --file DataExamples\clientCSP1.txt --name CSP1 --sep "Fiting started on Client..."
```

```sh
python LogParserFederated.py --file DataExamples\clientCSP2.txt --name CSP2 --sep "Fiting started on Client..."
```

```sh
python LogParserFederated.py --file DataExamples\clientCSP3.txt --name CSP3 --sep "Fiting started on Client..."
```

```sh
python LogParserUnified.py --file DataExamples\UnifiedCSP.txt --name CSP
```

```sh
python PlotterDriver.py --name1 CSP1 --name2 CSP2 --name3 CSP3 --nameU CSP --savedir CSP
```