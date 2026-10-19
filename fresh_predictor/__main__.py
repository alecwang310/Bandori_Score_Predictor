import argparse


def main():
    p=argparse.ArgumentParser(description='Fresh post-rule predictor training phases')
    p.add_argument('phase',choices=('select','assess','refit'));p.add_argument('--run',required=True)
    p.add_argument('--data',default='data/full_approaches_20261017')
    p.add_argument('--audit-events',type=int,nargs='+',help='For a new future protocol only: reserved audit event IDs')
    p.add_argument('--exclude-events',type=int,nargs='+',help='For a new future protocol only: forecast-only event IDs')
    a=p.parse_args()
    if a.phase!='select' and (a.audit_events or a.exclude_events):p.error('Split settings are sealed at selection; cannot change later')
    if a.phase=='select':
        from . import pipeline
        if a.audit_events:pipeline.AUDIT_IDS=tuple(a.audit_events)
        if a.exclude_events:pipeline.EXCLUDED=tuple(a.exclude_events)
        pipeline.select(a.data,a.run)
    elif a.phase=='assess':
        from .evaluation import assess
        assess(a.run)
    else:
        from .pipeline import refit
        refit(a.run)


if __name__=='__main__':main()
