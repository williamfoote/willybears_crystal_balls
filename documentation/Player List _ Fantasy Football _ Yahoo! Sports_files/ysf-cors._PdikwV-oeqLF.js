
YUI.add('ysf-cors',function(Y){var Cors=function(url,config){var corsIO=getCorsIOInstance(),xdrCfg={use:'native'};config=config?Y.Object(config):{};config.xdr=(Y.Lang.isObject(config.xdr))?Y.merge(xdrCfg,config.xdr):xdrCfg;return corsIO.send(url,config);},getCorsIOInstance=function(){var ioinstance;if(!Cors._io||!(Cors._io instanceof Y.IO)){if(!Y.io._map['io:0']){new Y.IO();}
ioinstance=new Y.IO();ioinstance._headers={};Cors._io=ioinstance;}
return Cors._io;};Y.namespace('Fantasy');Y.Fantasy.Cors=Cors;},'0.0.1',{requires:['io-base','io-xdr']});