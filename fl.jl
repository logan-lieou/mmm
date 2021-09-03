function clientUpdate(clientModel, optimizer, trainLoader, epoch=5)
   model.train()
   for e in 1:epoch
      for (data, target) in enumerate(train_loader)
         data, target = data.cuda(), target.cuda()
         optimizer = zero_grad()
         output = client_model(data)
         loss = F.nll_loss(output, target)
         loss.backward()
         optimizer.step()
      end
   end
   return loss.item()
end

function serverAggergate(globalModel, clientModels)
   globalDict = globalModel.stateDict() # TODO
   for k in globalDict.keys()
      globalDict[k] = torch.stack([clientModels[i].stateDict()[k].float() for i in range(len(clientModels))], 0).mean(0)
   end

   ## TODO .state_dict() in Flux?
   globalModel.load_state_dict(globalDict)
   for model in clientModels
      model.load_state_dict(global_model.state_dict())
   end
end

function test(globalMode, testLoader)
   # okay this is going to be totally diffrent in Flux
end

"""
def test(global_model, test_loader):
    # This function test the global model on test data and returns test loss and test accuracy
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc
"""