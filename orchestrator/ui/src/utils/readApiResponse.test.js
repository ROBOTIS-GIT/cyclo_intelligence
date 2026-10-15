import readApiResponse from './readApiResponse';
test.each([502,503,504])('retains status for HTML %s responses',async status=>{
 await expect(readApiResponse({ok:false,status,json:async()=>{throw new SyntaxError("Unexpected token '<'");}})).rejects.toMatchObject({status,message:expect.stringContaining(`HTTP ${status}`)});
});
test('preserves backend validation details',async()=>{
 await expect(readApiResponse({ok:false,status:409,json:async()=>({detail:'Model is loaded'})})).rejects.toMatchObject({status:409,message:'Model is loaded'});
});
test('passes through cancellation',async()=>{
 const error=Object.assign(new Error('cancelled'),{name:'AbortError'});
 await expect(readApiResponse({json:async()=>{throw error;}})).rejects.toBe(error);
});
