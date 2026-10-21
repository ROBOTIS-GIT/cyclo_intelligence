import { DaggerClient } from './daggerControl';
const mockTopics = [];
jest.mock('roslib', () => ({ Topic: function (args) {
  Object.assign(this,args); this.subscribe=jest.fn((fn)=>{this.handler=fn;});
  this.unsubscribe=jest.fn();this.unadvertise=jest.fn();this.publish=jest.fn();mockTopics.push(this);
}, Message: function (args) {Object.assign(this,args);} }));
beforeEach(()=>{mockTopics.length=0;});

test('commands cannot queue across a disconnect and ownership stays locked while status is stale',()=>{
  let now=1000;const ros={isConnected:true,callOnConnection:jest.fn()};
  const client=new DaggerClient(ros,()=>now);
  mockTopics[0].handler({data:JSON.stringify({state:'AUTO',busy:true})});
  client.command('hold');expect(mockTopics[1].publish).toHaveBeenCalledTimes(1);
  ros.isConnected=false;expect(()=>client.command('resume')).toThrow();
  mockTopics[1].callForSubscribeAndAdvertise({op:'publish'});expect(ros.callOnConnection).not.toHaveBeenCalled();
  now+=2000;expect(client.snapshot()).toMatchObject({online:false,busy:true});
});

test('malformed status never enables controls and dispose sends no robot command',()=>{
  const client=new DaggerClient({isConnected:true},()=>1000);
  mockTopics[0].handler({data:'broken'});expect(client.snapshot().online).toBe(false);
  client.dispose();expect(mockTopics[1].publish).not.toHaveBeenCalled();
});
