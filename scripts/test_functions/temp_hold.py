# bottom of DB field choice in pyspinalfmri
#def DBfieldchoice(self, value):

# save a copy of the covariates list for other analyses
p, f = os.path.split(self.DBname)
covsavename = os.path.join(p, 'copy_of_covariates.npy')
np.save(covsavename, {'GRPcharacteristicsvalues': self.GRPcharacteristicsvalues,
                      'GRPcharacteristicslist': self.GRPcharacteristicslist,
                      'GRPcharacteristicsvalues2': self.GRPcharacteristicsvalues2,
                      'GRPcharacteristicscount': self.GRPcharacteristicscount})