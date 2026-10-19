import argparse
import json
from pathlib import Path
import sys

from collect_bandori_data import timestamp
from .config import Config
from .dataset import Dataset
from .pipeline import train,predict
from .snapshot import snapshot_cache


def main(argv=None):
    parser=argparse.ArgumentParser(description='Compare two causal daily CN cutoff predictors')
    commands=parser.add_subparsers(dest='command',required=True)
    snapshot=commands.add_parser('snapshot');snapshot.add_argument('--source',default='bandori_cn_data');snapshot.add_argument('--output',required=True)
    fit=commands.add_parser('train');fit.add_argument('--data',required=True);fit.add_argument('--config',default='forecast_config.json')
    fit.add_argument('--output',required=True);fit.add_argument('--registry',default='evaluation_registry.json');fit.add_argument('--report')
    inference=commands.add_parser('predict');inference.add_argument('--data',required=True);inference.add_argument('--model',required=True)
    inference.add_argument('--event',required=True,type=int);inference.add_argument('--as-of',required=True);inference.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='snapshot':result=snapshot_cache(args.source,args.output)
        elif args.command=='train':
            config=Config.load(args.config);data=Dataset.load(args.data,config.tiers)
            report=train(data,config,args.output,args.registry)
            from .reporting import write_report,plot_report
            destination=Path(args.report) if args.report else Path('reports')/(Path(args.output).name+'.md')
            write_report(args.output,destination)
            try:plot_report(args.output,destination.with_suffix('.png'))
            except ImportError:print('Install matplotlib to generate plots.',file=sys.stderr)
            result={'selection':{k:{a:b for a,b in v.items() if a!='factors'} for k,v in report['selection'].items()},
                    'test':{k:v['overall'] for k,v in report['partitions']['test'].items()}}
        else:
            config=Config.load(Path(args.model)/'config.json');data=Dataset.load(args.data,config.tiers)
            cutoff=timestamp(args.as_of)
            if cutoff is None:raise ValueError('Use a timezone-qualified --as-of timestamp')
            result=predict(data,args.model,args.event,cutoff,args.output)
        print(json.dumps(result,indent=2));return 0
    except (ValueError,OSError,KeyError) as error:
        print('ERROR:',error,file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
